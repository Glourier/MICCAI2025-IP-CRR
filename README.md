# IP-CRR: Information Pursuit for Interpretable Classification of Chest Radiology Reports

### Abstract
The development of AI-based methods for analyzing radiology reports could lead to significant advances in medical diagnosis, from improving diagnostic accuracy, to enhancing efficiency and reducing workload. However, the lack of interpretability of existing AI-based methods has hindered their adoption in clinical settings. In this paper, we propose an interpretable-by-design framework for classifying radiology reports. The key idea is to extract a set of most informative facts from the report and use these facts to predict a diagnosis. The explanation for the prediction is the set of selected facts. We use the Information Pursuit framework to select an informative fact, the Flan-T5-large model to verify if the fact is present in the report, and a classifier to predict a disease from these facts. Experiments on the MIMIC-CXR dataset show the effectiveness of the proposed method, highlighting its potential to enhance trust and usability in medical AI. 




