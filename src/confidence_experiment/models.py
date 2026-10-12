import time

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .io import digest, write_json


def train_supervised(config, frames, labels, output, include_test=False, frozen=False):
    label_id = {label: i for i, label in enumerate(labels)}
    roles = [r for r in frames if include_test or r != "test"]
    y = np.array([label_id[x] for x in frames["fit"].label])
    methods = set(config["methods"])
    results, artifacts = {}, {}
    params = config.get("supervised", {})
    if "tfidf" in methods:
        start = time.perf_counter()
        if frozen:
            vectorizer, classifier = joblib.load(output / "tfidf.joblib")
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
            x = vectorizer.fit_transform(frames["fit"].text)
            classifier = LogisticRegression(C=params.get("C", 10), max_iter=params.get("max_iter", 1000), random_state=config["seed"])
            classifier.fit(x, y)
        results["tfidf"] = {}
        inference = {}
        for role in roles:
            tick = time.perf_counter()
            results["tfidf"][role] = classifier.decision_function(vectorizer.transform(frames[role].text))
            inference[role] = time.perf_counter() - tick
        if not frozen:
            joblib.dump((vectorizer, classifier), output / "tfidf.joblib")
        artifacts["tfidf"] = {"total_seconds": time.perf_counter() - start, "inference_seconds": inference}
    if methods & {"embedding_lr", "embedding_mlp"}:
        from huggingface_hub import model_info
        from sentence_transformers import SentenceTransformer
        settings = config["embedding"]
        from .io import read_json
        revision = (read_json(output / "embedding_model.json")["revision"] if frozen else
                    model_info(settings["model"], revision=settings.get("revision", "main")).sha)
        if not frozen:
            write_json(output / "embedding_model.json", {"model": settings["model"], "revision": revision})
        start = time.perf_counter()
        model = SentenceTransformer(settings["model"], revision=revision, device=settings.get("device", "cpu"))
        embedding = {}
        for role in set(roles) | {"fit"}:
            texts = frames[role].text.tolist()
            key = digest({"model": settings["model"], "revision": revision, "texts": texts, "prefix": "task: classification | query: "})
            path = output / "embeddings" / f"{key}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                embedding[role] = np.load(path)
            else:
                embedding[role] = model.encode(["task: classification | query: " + t for t in texts],
                    prompt="", batch_size=settings.get("batch_size", 16), normalize_embeddings=True, show_progress_bar=True)
                np.save(path, embedding[role])
        for name in sorted(methods & {"embedding_lr", "embedding_mlp"}):
            classifier = (LogisticRegression(C=params.get("C", 10), max_iter=params.get("max_iter", 1000), random_state=config["seed"])
                if name == "embedding_lr" else MLPClassifier(hidden_layer_sizes=(settings.get("hidden_units", 128),),
                    max_iter=settings.get("mlp_epochs", 100), random_state=config["seed"], early_stopping=False))
            if frozen:
                classifier = joblib.load(output / f"{name}.joblib")
            else:
                classifier.fit(embedding["fit"], y)
            results[name] = {}
            for role in roles:
                if name == "embedding_lr":
                    results[name][role] = classifier.decision_function(embedding[role])
                else:
                    results[name][role] = np.log(np.clip(classifier.predict_proba(embedding[role]), 1e-30, 1))
            if not frozen:
                joblib.dump(classifier, output / f"{name}.joblib")
            artifacts[name] = {"model": settings["model"], "revision": revision, "total_seconds_shared_embeddings": time.perf_counter() - start}
        # Inference-time signals for correctness estimation, reference labels only from fit.
        from sklearn.neighbors import NearestNeighbors
        reference = NearestNeighbors(n_neighbors=5, metric="cosine").fit(embedding["fit"])
        artifacts["embedding_features"] = {}
        for role in roles:
            distances, indices = reference.kneighbors(embedding[role])
            artifacts["embedding_features"][role] = (distances[:, 0], y[indices])
    if "modernbert" in methods:
        logits, metadata = train_encoder(config, frames, label_id, roles, output, frozen=frozen)
        results["modernbert"] = logits
        artifacts["modernbert"] = metadata
    return results, artifacts


def train_encoder(config, frames, label_id, roles, output, frozen=False):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed
    from huggingface_hub import model_info

    settings = config["encoder"]
    set_seed(config["seed"])
    from .io import read_json
    revision = (read_json(output / "encoder_model.json")["revision"] if frozen else
                model_info(settings["model"], revision=settings.get("revision", "main")).sha)
    source = output / "modernbert" if frozen else settings["model"]
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision if not frozen else None)
    model = AutoModelForSequenceClassification.from_pretrained(source, revision=revision if not frozen else None,
        num_labels=len(label_id), id2label={v: k for k, v in label_id.items()}, label2id=label_id,
        attn_implementation="eager")
    device = torch.device(settings.get("device", "cpu"))
    model.to(device)

    class TextDataset(Dataset):
        def __init__(self, frame):
            self.tokens = tokenizer(frame.text.tolist(), truncation=True, max_length=settings.get("max_length", 256))
            self.labels = [label_id[label] for label in frame.label]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {**{key: value[i] for key, value in self.tokens.items()}, "labels": self.labels[i]}

    collator = DataCollatorWithPadding(tokenizer)
    batch_size = settings.get("batch_size", 8)
    loader = DataLoader(TextDataset(frames["fit"]), batch_size=batch_size, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get("learning_rate", 2e-5))
    start = time.perf_counter()
    model.train()
    for epoch in range(0 if frozen else settings.get("epochs", 3)):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"ModernBERT epoch {epoch + 1}: loss={total / len(loader):.4f}", flush=True)
    if not frozen:
        model.save_pretrained(output / "modernbert")
        tokenizer.save_pretrained(output / "modernbert")
        write_json(output / "encoder_model.json", {"model": settings["model"], "revision": revision})
    result = {}
    model.eval()
    with torch.no_grad():
        for role in roles:
            loader = DataLoader(TextDataset(frames[role]), batch_size=batch_size, collate_fn=collator)
            result[role] = np.concatenate([model(**{k: v.to(device) for k, v in batch.items() if k != "labels"}).logits.cpu().numpy() for batch in loader])
    return result, {"revision": revision, "total_seconds": time.perf_counter() - start}
