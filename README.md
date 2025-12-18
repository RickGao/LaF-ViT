# LaF-ViT: Layered Fairness Vision Transformer

LaF-ViT is a hierarchical multi-task Vision Transformer for facial attribute estimation. It jointly predicts **gender**, **race**, and **age**, modeling **age estimation as a downstream task** softly conditioned on demographic predictions to improve robustness and fairness.

![LaF-ViT Architecture](asset/arch.png)

## Key Features
- **Hierarchical multi-task learning** with age conditioned on gender/race
- **Soft conditioning** using probability distributions (not hard labels)

## Results
Evaluated on **UTKFace**, LaF-ViT reduces age estimation error vs. a ResNet-50 baseline, while maintaining comparable or improved gender and race accuracy, with lower error variance across demographic groups.

## Dataset
- UTKFace
