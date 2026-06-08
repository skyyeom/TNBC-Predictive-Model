# Literature Review

## Multi-omic pCR Prediction

Sammut et al. showed that breast cancer therapy response can be predicted by integrating clinical, digital pathology, genomic, and transcriptomic features. The key modeling insight was not a single complex model, but a carefully structured pipeline: feature curation, univariable selection, collinearity reduction, and an unweighted ensemble of logistic regression, SVM, and random forest models.

## TNBC and pCR

TNBC is clinically important because it lacks ER, PR, and HER2 targets, making chemotherapy response a central determinant of treatment planning. pCR is a meaningful endpoint in neoadjuvant TNBC studies because it reflects eradication of invasive disease in breast and sampled nodes after therapy.

## MRI Prediction of pCR

Breast DCE-MRI provides an opportunity to observe tumor burden and vascular response before, during, and after therapy. Longitudinal MRI features, such as tumor volume reduction, longest diameter change, enhancement kinetics, and ADC changes, can capture early treatment response that clinical features alone miss.

## MRI Segmentation

Segmentation is important because radiomics and deep features should be localized to tumor tissue rather than the full breast volume. nnU-Net provides a strong self-configuring baseline for medical segmentation. Swin UNETR and related transformer models provide stronger long-range representation learning. Medical adaptations of Segment Anything can also support annotation-efficient workflows.

## Teacher-Student Learning

Knowledge distillation transfers behavior from a large teacher model to a smaller student model. In this project, distillation is used to produce a deployable tumor segmentation model that can generate consistent tumor masks across heterogeneous public MRI datasets.
