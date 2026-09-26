"""Multi-task ModernBERT risk classifier for inferential privacy detection.

Predicts binary inferability for age, location, occupation, and education.
Provides input embedding hooks for Integrated Gradients and temperature scaling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

ATTRIBUTES: Tuple[str, ...] = ("age", "location", "occupation", "education")
ATTR_TO_IDX: Dict[str, int] = {attr: idx for idx, attr in enumerate(ATTRIBUTES)}
IDX_TO_ATTR: Dict[int, str] = {idx: attr for idx, attr in enumerate(ATTRIBUTES)}


class ModernBertRiskClassifier(nn.Module):
    """4-head multi-task risk classifier based on ModernBERT-base."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_heads: int = 4,
        num_classes_per_head: int = 1,
        dropout_prob: float = 0.1,
        attn_implementation: str = "sdpa",
        backbone: Optional[nn.Module] = None,
        hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head

        if backbone is not None:
            self.bert = backbone
            resolved_hidden_size = hidden_size or getattr(
                backbone, "hidden_size", getattr(getattr(backbone, "config", None), "hidden_size", 768)
            )
            self.config = getattr(backbone, "config", None)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            try:
                self.bert = AutoModel.from_pretrained(
                    model_name,
                    config=self.config,
                    attn_implementation=attn_implementation,
                )
            except Exception:
                self.bert = AutoModel.from_pretrained(model_name, config=self.config)
            resolved_hidden_size = getattr(self.config, "hidden_size", 768)

        effective_dropout = (
            getattr(self.config, "classifier_dropout", None)
            or getattr(self.config, "hidden_dropout_prob", None)
            or dropout_prob
        )
        self.dropout = nn.Dropout(effective_dropout)

        self.classifiers = nn.ModuleList(
            [nn.Linear(resolved_hidden_size, num_classes_per_head) for _ in range(num_heads)]
        )

        self.temperatures = nn.ParameterList(
            [nn.Parameter(torch.ones(1), requires_grad=False) for _ in range(num_heads)]
        )

    def get_input_embeddings(self) -> Optional[nn.Embedding]:
        """Return the backbone input embedding layer for Integrated Gradients."""
        if hasattr(self.bert, "get_input_embeddings"):
            emb = self.bert.get_input_embeddings()
            if emb is not None:
                return emb
        if hasattr(self.bert, "embeddings"):
            if hasattr(self.bert.embeddings, "tok_embeddings"):
                return self.bert.embeddings.tok_embeddings
            return self.bert.embeddings
        return None

    def set_temperatures(self, temperatures: Sequence[float]) -> None:
        """Set calibrated temperatures for all heads."""
        if len(temperatures) != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} temperatures, got {len(temperatures)}")
        for i, t in enumerate(temperatures):
            if t <= 0.0:
                raise ValueError(f"Temperature must be strictly positive, got {t} for head {i}")
            self.temperatures[i].data = torch.tensor([float(t)], dtype=torch.float32)

    def get_temperatures(self) -> List[float]:
        """Return list of current temperature values."""
        return [round(float(p.item()), 6) for p in self.temperatures]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        apply_temperature: bool = False,
    ) -> List[torch.Tensor]:
        """Compute logits for all 4 heads.

        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_len].
            attention_mask: Mask tensor of shape [batch_size, seq_len].
            inputs_embeds: Embedding tensor of shape [batch_size, seq_len, hidden_size].
            apply_temperature: Whether to divide raw logits by head temperatures.

        Returns:
            List of 4 tensors, each of shape [batch_size, num_classes_per_head].
        """
        if (input_ids is None and inputs_embeds is None) or (
            input_ids is not None and inputs_embeds is not None
        ):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            )

        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            sequence_output = outputs
        else:
            raise TypeError(f"Unexpected output type from backbone: {type(outputs)}")

        cls_representation = sequence_output[:, 0, :]
        cls_representation = self.dropout(cls_representation)

        logits: List[torch.Tensor] = []
        for i, classifier in enumerate(self.classifiers):
            head_logits = classifier(cls_representation)
            if apply_temperature:
                head_logits = head_logits / self.temperatures[i]
            logits.append(head_logits)

        return logits

    def predict_proba(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        apply_temperature: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute sigmoid probability tensor [batch_size, 1] per attribute."""
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            apply_temperature=apply_temperature,
        )
        probs: Dict[str, torch.Tensor] = {}
        for i, attr in enumerate(ATTRIBUTES):
            head_logit = logits[i]
            if head_logit.shape[-1] == 1:
                probs[attr] = torch.sigmoid(head_logit)
            else:
                probs[attr] = torch.softmax(head_logit, dim=-1)[:, 1:2]
        return probs

    def save_pretrained(self, save_dir: Union[str, Path]) -> None:
        """Save model weights and architecture configuration."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "model.pt")
        config_data = {
            "model_name": self.model_name,
            "num_heads": self.num_heads,
            "num_classes_per_head": self.num_classes_per_head,
            "attributes": list(ATTRIBUTES),
            "temperatures": self.get_temperatures(),
        }
        with open(save_path / "risk_model_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_dir: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        backbone: Optional[nn.Module] = None,
    ) -> ModernBertRiskClassifier:
        """Instantiate model and load serialized weights and temperatures."""
        load_path = Path(load_dir)
        with open(load_path / "risk_model_config.json", "r", encoding="utf-8") as f:
            config_data = json.load(f)

        model = cls(
            model_name=config_data["model_name"],
            num_heads=config_data["num_heads"],
            num_classes_per_head=config_data["num_classes_per_head"],
            backbone=backbone,
        )
        state_dict = torch.load(load_path / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        if "temperatures" in config_data:
            model.set_temperatures(config_data["temperatures"])
        model.to(device)
        return model
