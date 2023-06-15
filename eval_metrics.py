import monai
import torch

def compute_metric(dataloader, model, metric_fn, device):
    """
    This function computes the average value of a metric for a data set.

    Args:
        dataloader (monai.data.DataLoader): dataloader wrapping the dataset to evaluate.
        model (torch.nn.Module): trained model to evaluate.
        metric_fn (function): function computing the metric value from two tensors:
            - a batch of outputs,
            - the corresponding batch of ground truth masks.

    Returns:
        (float) the mean value of the metric
    """
    model.eval()
    inferer = monai.inferers.SlidingWindowInferer(roi_size=[200, 200])
    discrete_transform = monai.transforms.AsDiscrete(threshold=0.5)
    Sigmoid = torch.nn.Sigmoid()

    for sample in dataloader:
        with torch.no_grad():
            output_class1 = discrete_transform(Sigmoid(inferer(sample['img'].to(device), network=model).cpu()))

        batch_class_dice = metric_fn(y_pred=output_class1, y=sample["mask"], include_background=False)

    return batch_class_dice, batch_class_dice.mean(dim=0)