
import torch
from nn_bowl.costs.cost import Cost

class CrossEntropyErrorCost(Cost):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.metrics["accuracy"] = []
        self.lam = cfg.costs.errors.lam

    """
    CrossEntropyErrorCost calculates the cross-entropy loss between model predictions and true labels.
    """
    
    def forward(self, out, labels):
        accuracy = torch.sum(out["probs"].argmax(dim=1) == labels).detach() / labels.size(0) * 100.0
        self.metrics["accuracy"].append(accuracy)

        # adapted to use the mixture of probabilities instead of logits
        log_probs = torch.log(out["probs"] + 1e-8)
        error_cost = torch.nn.functional.nll_loss(log_probs, labels)

        # normalizing by log(num_classes) to keep it in a similar range as other costs (0-1) and across datasets with different number of classes
        error_cost_normalized = error_cost / torch.log(torch.tensor(out["probs"].shape[1], dtype=torch.float32))

        return self.lam * error_cost_normalized



