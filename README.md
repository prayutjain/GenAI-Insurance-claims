
# Medical Claim Data Processing Project

This project encompasses a comprehensive pipeline for processing medical claim data. It includes the generation of artificial forms, extraction of key information using a trained BERT model, and summarization of extracted data with a focus on prescription details. The project leverages both AI models and custom data processing techniques.

## Components

### 1. Artificial Form Generation
- **Patient Persona Generation**: Uses random value generators to create diverse patient personas.
- **Physician, Diagnosis, Prescription Data Generation**: Utilizes OpenAI's GPT-4 API to generate realistic physician info, diagnoses, and prescriptions. The GPT-4 API complements the random value generators for a more efficient and cost-effective data generation process.

### 2. Information Extraction
- **Model Training**: A BERT model (`bert-base-uncased`) is trained for Named Entity Recognition (NER) on 80 annotated forms.
- **Data Annotation**: Annotations were created using the GPT-4 API for various tags like Aetna Number, Patient Name, Date of Birth, Drug, Dosage, etc.
- **Training Details**: The model was fine-tuned using the AdamW optimizer, with a learning rate of 5x10^-5 over 30 epochs.
- **Output**: The model outputs predictions in JSON format, facilitating easy integration and further processing.

### 3. Summarization
- **Approach**: Utilizes the trained BERT model for summarizing extracted information, focusing on drug-related data such as drug names, dosages, and frequencies.
- **Methodology**:
  - **Method 1**: An initial approach using our pretrained model, which provided baseline results.
  - **Method 2**: Leveraged a pretrained clinical model, "Posos/ClinicalNER", from Hugging Face, which demonstrated improved accuracy in identifying drugs and dosages.

## Project Structure
- **Main.ipynb**: The main Jupyter notebook where all processes are executed.
- **Helpers.py**: Contains helper classes and functions for various tasks like form generation, data annotation, and prediction.
- **training_dataset.csv**: Dataset comprising annotated data from 80 forms, used for training the BERT model.
- **ner_model**: Folder containing the trained BERT model.
- **sample_forms**: A collection of artificially generated sample forms for reference.

## Usage
- To run the project, start with `Main.ipynb`, which will guide you through the entire process step by step.
- Ensure all dependencies are installed as per the requirements listed in `Helpers.py`.
- The BERT model and the "Posos/ClinicalNER" model can be loaded as described in their respective sections.

---

This README provides an overview of the medical data processing project, covering its key components, project structure, and usage instructions.
