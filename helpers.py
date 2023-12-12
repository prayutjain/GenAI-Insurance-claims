import random
from pypdf import PdfReader, PdfWriter
from openai import OpenAI
import ast
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
import pandas as pd
import json
from transformers import AutoModelForTokenClassification, AutoTokenizer


os.environ['OPENAI_API_KEY'] = ''

client = OpenAI()

class FormGenerator:
    def __init__(self):
        pass

    
    @staticmethod
    def random_member_number():
        return ''.join([str(random.randint(0, 9)) for _ in range(14)])

    @staticmethod
    def random_birth_date():
        year = random.randint(1950, 2003)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{month:02d}/{day:02d}/{year}"

    @staticmethod
    def random_recent_date():
        year = 2023
        month = random.randint(1, 11)
        day = random.randint(1, 28)
        return f"{month:02d}/{day:02d}/{year}"
    
    @staticmethod
    def random_lastyear_date():
        year = 2022
        month = random.randint(1, 11)
        day = random.randint(1, 28)
        return f"{month:02d}/{day:02d}/{year}"

    @staticmethod
    def random_phone():
        return f"{random.randint(100, 999)}-{random.randint(1000000, 9999999)}"

    @staticmethod
    def random_yes_no():
        return random.choice(["/Yes", "/No"])

    @staticmethod
    def random_gender():
        return random.choice(["/male", "/female"])

    @staticmethod
    def generate_random_name():
        first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]

        first_name = random.choice(first_names)
        last_name = random.choice(last_names)

        return f"{first_name} {last_name}"

    @staticmethod
    def generate_random_address():
        streets = ['Main St', 'Elm St', 'Maple Ave', 'Oak St', 'Pine St']
        cities = ['New York', 'Chicago', 'Los Angeles', 'Houston', 'Phoenix']
        states = ['NY', 'IL', 'CA', 'TX', 'AZ']
        zip_codes = ['10001', '60601', '90001', '77001', '85001']
        
        street = random.choice(streets)
        city = random.choice(cities)
        state = random.choice(states)
        zip_code = random.choice(zip_codes)
        return f"{random.randint(100, 999)} {street}, {city}, {state} - {zip_code}"

    @staticmethod
    def generate_random_company():
        company_names = ['TechWave Solutions', 'GreenLeaf Innovations', 'Quantum Dynamics', 'BrightBridge Technologies', 'Visionary Analytics']
        streets = ['Innovation Way', 'Enterprise Ave', 'Technology Blvd', 'Industrial Pkwy', 'Commerce St']
        cities = ['Techville', 'Innovatown', 'Enterprisecity', 'Industriopolis', 'Commerceburg']
        states = ['UT', 'CA', 'TX', 'MA', 'NY']
        zip_codes = ['84001', '94000', '75001', '02101', '10001']

        company_name = random.choice(company_names)
        street = random.choice(streets)
        city = random.choice(cities)
        state = random.choice(states)
        zip_code = random.choice(zip_codes)
        address = f"{random.randint(100, 999)} {street}, {city}, {state} - {zip_code}"
        
        return f"{company_name}, {address}"

    def generate_gpt_response(self, prompt):
        
        empty_dict = {
            'Physician name and address': None,
            'Facility': None,
            'Diagnosis_1': None,
            'Diagnosis_2': None,
            'Description_1': None,
            'Description_2': None,
            'Description_3': None,
            }
        try:
            # Assuming client is correctly initialized as an OpenAI client
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3,
                    top_p=0.5
                )
            except openai.error.OpenAIError as e:
                print(f"OpenAI API error: {e}")
                return empty_dict
            except Exception as e:
                print(f"Error making API call: {e}")
                return empty_dict

            if not response.choices or not response.choices[0].message:
                print("Invalid response structure from GPT API.")
                return empty_dict

            response_text = response.choices[0].message.content

            # Parsing the response to a dictionary
            try:
                response_dict = ast.literal_eval(response_text)
                if not isinstance(response_dict, dict):
                    raise ValueError("Response is not a dictionary")

            except (ValueError, SyntaxError) as e:
                print(f"Error parsing GPT response to dict: {e}")
                return empty_dict

            return response_dict

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return empty_dict

    def extract_readable_form_field_names(self, input_pdf_path):
        with open(input_pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            fields = pdf_reader.get_fields()

            readable_names = {}
            if fields:
                for field_name, field_info in fields.items():
                    readable_name = field_info.get('/TU', 'No label found')
                    readable_names[field_name] = readable_name

                return readable_names
            else:
                return "No fillable form fields found."

    ## Commercial prescription drug claim form
    def field_values_to_fill_cpdcf(self, input_pdf_path):

        field_names = self.extract_readable_form_field_names(input_pdf_path)
        
        ## All prescriptions for self
        name = self.generate_random_name()
        birthdate = self.random_birth_date()

        if not field_names == "No fillable form fields found.":

            if random.random() < 0.2:
                field_values = {'Aetna Member Number (claim cannot be processed without number)': self.random_member_number(),
                    'Group Number': self.random_member_number(),
                    'If you are enrolled in Medicare, check here': self.random_yes_no(),
                    'Employee Name (First, Middle, Last)': name,
                    'Employee Birthdate (MM/DD/YYYY)': birthdate,
                    'Employee Address (Street, City, State, ZIP Code)': self.generate_random_address(),
                    'Company Name & Address (Street, City, State, ZIP Code)': self.generate_random_company(),
                    'Date (MM/DD/YYYY)': self.random_recent_date(),
                    'Last Name, First, Middle Initial': name,
                    'Patient Birthdate (MM/DD/YYYY)': birthdate,
                    'Gender': self.random_gender(),
                    'Employee, Spouse, Dependent': '/employee',
                    'Emergency – If Emergency, describe Emergency below, or on a separate sheet.': self.random_yes_no()
                    }
            else:
                field_values = {'Aetna Member Number (claim cannot be processed without number)': self.random_member_number(),
                    'Group Number': self.random_member_number(),
                    'If you are enrolled in Medicare, check here': self.random_yes_no(),
                    'Employee Name (First, Middle, Last)': self.generate_random_name(),
                    'Employee Birthdate (MM/DD/YYYY)': self.random_birth_date(),
                    'Employee Address (Street, City, State, ZIP Code)': self.generate_random_address(),
                    'Company Name & Address (Street, City, State, ZIP Code)': self.generate_random_company(),
                    'Date (MM/DD/YYYY)': self.random_recent_date(),
                    'Last Name, First, Middle Initial': self.generate_random_name(),
                    'Patient Birthdate (MM/DD/YYYY)': self.random_birth_date(),
                    'Gender': self.random_gender(),
                    'Employee, Spouse, Dependent': random.choice(["/spouse", "/dependent"]),
                    'Emergency – If Emergency, describe Emergency below, or on a separate sheet.': self.random_yes_no()
                    }
            
            merged_dict = {key: field_values.get(value, '') for key, value in field_names.items()}
            final_dict = {k: v for k, v in merged_dict.items() if v and v != 'None'}
            final_dict

        else:
            final_dict = "No fillable form fields found."

        return final_dict
    
    ## Medical benefits form
    def field_values_to_fill_mbr(self, input_pdf_path):

        field_names = self.extract_readable_form_field_names(input_pdf_path)
        retirement = {'Active': '/No', 'Retired': '/Yes'} if random.choice(["/Yes", "/No"]) == "/Yes" else {'Active': '/Yes', 'Retired': '/No'}
        gender = {'Male': '/Yes', 'Female': '/No'} if random.choice(["/Yes", "/No"]) == "/Yes" else {'Male': '/No', 'Female': '/Yes'}
        marital = {'Married': '/Yes', 'Single': '/No'} if random.choice(["/Yes", "/No"]) == "/Yes" else {'Married': '/No', 'Single': '/Yes'}
        date = self.random_recent_date()


        field_values = {"Employer's Name": random.choice(['TechWave Solutions', 'GreenLeaf Innovations', 'Quantum Dynamics', 'BrightBridge Technologies', 'Visionary Analytics']),
                        "Employee's Aetna ID Number": self.random_member_number(),
                        'Policy/Group Number': self.random_member_number(),
                        "Empoloyee's Name": self.generate_random_name(),
                        "Employee's Birthdate (MM/DD/YYYY)": self.random_birth_date(),
                        'Active': retirement['Active'],
                        'Retired': retirement['Retired'],
                        "Employee's Address (include ZIP Code)": self.generate_random_address(),
                        "Patient's Name": self.generate_random_name(),
                        "Patient's Aetna ID Number": self.random_member_number(),
                        "Patient's Birthdate (MM/DD/YYYY)": self.random_birth_date(),
                        'Other': '/Yes',
                        "Patient's Address (if different from employee)": random.choice([self.generate_random_address(), None]),
                        'Male':gender['Male'],
                        'Female':gender['Female'],
                        'Married':marital['Married'],
                        'Single':marital['Single'],
                        'If claim is related to medical services received outside of the U.S, what is the name of the country were you received services?':random.choice(['Canada','India','Australia','Mexico','United Kingdom',None]),
                        'Date of Illness (first symptom) or injury (accident) or pregnancy (LMP) (MM/DD/YYYY)': date,
                        'Date first consulted you for this condition (MM/DD/YYYY)': date,
                        'If patient has had similar illness or injury, give dates (MM/DD/YYYY)': random.choice([None,self.random_lastyear_date()]),
                        'emergency': self.random_yes_no(),
                        'Patient Account Number': ''.join([str(random.randint(0, 9)) for _ in range(9)]),
                        'National Provider Identifier': ''.join([str(random.randint(0, 9)) for _ in range(10)]),
                        'Date (MM/DD/YYYY)': self.random_recent_date()
                        }
        
        merged_dict = {key: field_values.get(value, '') for key, value in field_names.items()}

        diseases = ["Hypertension", "Type 2 Diabetes", "Asthma", "High Cholesterol", "Depression", 
            "Gastroesophageal Reflux Disease (GERD)", "Hypothyroidism", "Arthritis", "Anxiety", "Migraine"]

        prompt = (
            "Generate a realistic and detailed medical scenario including a physician's name and address, "
            "the name and address of a medical facility, a primary and secondary diagnosis chosen randomly from "
            f"{random.sample(diseases, 2)}, and prescribe three medications including their dosage and frequency "
            "for those conditions. Format the information as follows:\n\n"
            "Response Format:\n"
            "{\n"
            "    'Physician name and address': 'Dr. [Physician's Full Name]\r[Street Adrress]\r[Apt./Suite]\r[city, state, zipcode]',\n"
            "    'Facility': '[Facility Name], [Facility Address]',\n"
            "    'Diagnosis_1': 'Primary Diagnosis: [Primary Illness]',\n"
            "    'Diagnosis_2': 'Secondary Diagnosis: [Secondary Illness]',\n"
            "    'Description_1': '[Drug 1 Name], [Dosage Amount], [Frequency]',\n"
            "    'Description_2': '[Drug 2 Name], [Dosage Amount], [Frequency]'\n"
            "    'Description_3': '[Drug 3 Name], [Dosage Amount], [Frequency]'\n"
            "}\n\n"
            "The response must strictly adhere to the above format as a dictionary. Each element should be "
            "filled with appropriate and realistic information, suited for a hypothetical medical case."
        )


        gpt_response = self.generate_gpt_response(prompt)
        merged_dict['Text100'] = gpt_response['Physician name and address']
        merged_dict['Text63'] = random.choice([None, gpt_response['Facility']])
        merged_dict['Text64'] = gpt_response['Diagnosis_1']
        merged_dict['Text65'] = gpt_response['Diagnosis_2']
        
        ## Procedure
        merged_dict['Text68'] = date
        merged_dict['Text71'] = gpt_response['Description_1']
        amt1 = random.randint(1000, 5000)
        merged_dict['Text73'] = amt1
        merged_dict['Text74'] = random.randint(1, 30)

        merged_dict['Text76'] = date
        merged_dict['Text79'] = gpt_response['Description_2']
        amt2 = random.randint(1000, 5000)
        merged_dict['Text81'] = amt2
        merged_dict['Text82'] = random.randint(1, 30)

        merged_dict['Text84'] = date
        merged_dict['Text87'] = gpt_response['Description_3']
        amt3 = random.randint(1000, 5000)
        merged_dict['Text89'] = amt3
        merged_dict['Text90'] = random.randint(1, 30)

        merged_dict['Text105'] = f'{amt1 + amt2 + amt3}'
        merged_dict['Text106'] = f'{amt1 + amt2 + amt3}'
        merged_dict['Text107'] = '0'
        
        final_dict = {k: v for k, v in merged_dict.items() if v and v != 'None'}
        final_dict

        return final_dict
    
    def field_values_to_fill_forma(self, input_pdf_path):

        field_names = self.extract_readable_form_field_names(input_pdf_path)
        date = self.random_recent_date()


        field_values = {"Enter patient's first name.": self.generate_random_name(),
                        'Enter insured ID number.': self.random_member_number(),
                        "Enter patient's telephone number (include area code).": self.random_phone(),
                        "Enter insured's telephone number (include area code)." : self.random_phone(),
                        "Enter insured's first name.": self.generate_random_name(),
                        "Choose patient's relationship to insured. Select this field for other.": "/Yes",
                        "Enter insured's date of birth  (MM/DD/YYYY).": self.random_birth_date(),
                        "Enter insured's policy group or FECA number.": self.random_member_number(),
                        'Enter date (MM/DD/YYYY).': date,
                        "Enter patient's street address.": self.generate_random_address(),
                        "Enter insured's street address.": self.generate_random_address(),
                        'Enter date (MM/DD/YYYY) of current illness, injury, or pregnancy.': self.random_recent_date(),
                        "Enter patient's birth date (MM/DD/YYYY).": self.random_birth_date(),
                        'Enter federal tax ID number.': ''.join([str(random.randint(0, 9)) for _ in range(6)]),
                        "Enter patient's account number." : ''.join([str(random.randint(0, 9)) for _ in range(9)])
                        }
        
        merged_dict = {key: field_values.get(value, '') for key, value in field_names.items()}

        prompt = (
            "Generate a realistic and detailed medical scenario including a physician's name and address, "
            "the name and address of a medical facility, a random primary and secondary diagnosis (such as COVID, pneumonia, dengue, Asthma, Depression), "
            "and prescribe three medications including their dosage and frequency. "
            "Format the information as follows:\n\n"
            "Response Format:\n"
            "{\n"
            "    'Physician name and address': 'Dr. [Physician's Full Name]\r[Street Adrress]\r[Apt./Suite]\r[city, state, zipcode]',\n"
            "    'Facility': '[Facility Name], [Facility Address]',\n"
            "    'Diagnosis_1': 'Primary Diagnosis: [Primary Illness]',\n"
            "    'Diagnosis_2': 'Secondary Diagnosis: [Secondary Illness]',\n"
            "    'Description_1': '[Drug 1 Name], [Dosage Amount], [Frequency]',\n"
            "    'Description_2': '[Drug 2 Name], [Dosage Amount], [Frequency]'\n"
            "    'Description_3': '[Drug 3 Name], [Dosage Amount], [Frequency]'\n"
            "}\n\n"
            "The response must strictly adhere to the above format as a dictionary. Each element should be "
            "filled with appropriate and realistic information, suited for a hypothetical medical case."
        )

        npi = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        gpt_response = self.generate_gpt_response(prompt)
        merged_dict['owcp1500[0].page1[0].billingProvider[0].billingProvider[0].textField[0]'] = gpt_response['Physician name and address']
        merged_dict['owcp1500[0].page1[0].serviceFacility[0].serviceFacility[0].textField[0]'] = gpt_response['Facility']
        merged_dict['owcp1500[0].page1[0].diagnosis[0].af[0]'] = gpt_response['Diagnosis_1']
        merged_dict['owcp1500[0].page1[0].diagnosis[0].ef[0]'] = gpt_response['Diagnosis_2']
        
        ## Procedure
        merged_dict['owcp1500[0].page1[0].section24[0].Row1_Section24[0].datesofService[0].dateField1[0]'] = date
        merged_dict['owcp1500[0].page1[0].section24[0].Row1_Section24[0].datesofService[0].dateField2[0]'] = date
        merged_dict['owcp1500[0].page1[0].section24[0].Row1_Section24[0].rendering[0].rendering_1a[0]'] = npi
        merged_dict['owcp1500[0].page1[0].section24[0].Row1_Section24[0].CPT[0].label[0]'] = 'Level II'
        merged_dict['owcp1500[0].page1[0].section24[0].Row1_Section24[0].modifier[0].label[0]'] = gpt_response['Description_1']
        amt1 = random.randint(1000, 5000)
        merged_dict['owcp1500[0].page1[0].section24[0].Row1_Section24[0].charges[0].label[0]'] = amt1
        merged_dict['owcp1500[0].page1[0].section24[0].Row1_Section24[0].days[0].label[0]'] = random.randint(1, 30)

        merged_dict['owcp1500[0].page1[0].section24[0].Row2_Section24[0].datesofService[0].dateField1[0]'] = date
        merged_dict['owcp1500[0].page1[0].section24[0].Row2_Section24[0].datesofService[0].dateField2[0]'] = date
        merged_dict['owcp1500[0].page1[0].section24[0].Row2_Section24[0].rendering[0].rendering_1a[0]'] = npi
        merged_dict['owcp1500[0].page1[0].section24[0].Row2_Section24[0].CPT[0].label[0]'] = 'Level II'
        merged_dict['owcp1500[0].page1[0].section24[0].Row2_Section24[0].modifier[0].label[0]'] = gpt_response['Description_2']
        amt2 = random.randint(1000, 5000)
        merged_dict['owcp1500[0].page1[0].section24[0].Row2_Section24[0].charges[0].label[0]'] = amt2
        merged_dict['owcp1500[0].page1[0].section24[0].Row2_Section24[0].days[0].label[0]'] = random.randint(1, 30)

        merged_dict['owcp1500[0].page1[0].section24[0].Row3_Section24[0].datesofService[0].dateField1[0]'] = date
        merged_dict['owcp1500[0].page1[0].section24[0].Row3_Section24[0].datesofService[0].dateField2[0]'] = date
        merged_dict['owcp1500[0].page1[0].section24[0].Row3_Section24[0].rendering[0].rendering_1a[0]'] = npi
        merged_dict['owcp1500[0].page1[0].section24[0].Row3_Section24[0].CPT[0].label[0]'] = 'Level II'
        merged_dict['owcp1500[0].page1[0].section24[0].Row3_Section24[0].modifier[0].label[0]'] = gpt_response['Description_3']
        amt3 = random.randint(1000, 5000)
        merged_dict['owcp1500[0].page1[0].section24[0].Row3_Section24[0].charges[0].label[0]'] = amt3
        merged_dict['owcp1500[0].page1[0].section24[0].Row3_Section24[0].days[0].label[0]'] = random.randint(1, 30)

        merged_dict['owcp1500[0].page1[0].totalCharge[0].charge1[0]'] = f'{amt1 + amt2 + amt3}'
        merged_dict['owcp1500[0].page1[0].amountPaid[0].charge1[0]'] = f'{amt1 + amt2 + amt3}'
        
        final_dict = {k: v for k, v in merged_dict.items() if v and v != 'None'}
        final_dict

        return final_dict

    def write_form(self, input_pdf_path, output_pdf_path,page, field_dict):

        reader = PdfReader(input_pdf_path)
        writer = PdfWriter()

        writer.append(reader)
        # field_dict = self.field_values_to_fill()

        if not field_dict == "No fillable form fields found.":

            writer.update_page_form_field_values(
                writer.pages[page], field_dict
            )

            # write "output" to pypdf-output.pdf
            with open(output_pdf_path, "wb") as output_stream:
                writer.write(output_stream)

        else:
            print("No fillable form fields found.")


class annotate_data:

    def __init__(self):
        pass

    def extract_readable_form_field_names(self, pdf_file_path):
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            fields = pdf_reader.get_fields()

            readable_names = {}
            if fields:
                for field_name, field_info in fields.items():
                    readable_name = field_info.get('/TU', 'No label found')
                    value = field_info.get('/V', None)
                    readable_names[field_name] = [readable_name, value]

                return readable_names
            else:
                return "No fillable form fields found."

    def generate_annotation(self, input_text):

        prompt = (
            f'Annotate the provided input text for training a BERT model, focusing on extracting specific information categories: '
            f'AETNA id, patient name, patient birthdate, gender, hospital name, physician name, diagnosis, drug, dosage, frequency and amount paid. The annotation should follow the format shown in this example:\n'
            f'\t- Example: \n'
            f'\t\tText: "The patient, John Doe, was prescribed Aspirin, 100mg twice daily."\n'
            f'\t\tAnnotations: {{"entities": [(16, 24, "PATIENT_NAME"), (40, 47, "DRUG"), (49, 54, "DOSAGE"), (55, 66, "FREQUENCY")]}}\n'
            f'Given input text:\n'
            f'\t- Input Text: "{input_text}"\n'
            f'Provide the annotations in the same format as the example ONLY, identifying and labeling the relevant entities in the input text.'
        )
        response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.3,
                        top_p=0.5
                    )
        
        gpt_response = response.choices[0].message.content

        # Using regex to extract text within the outermost curly braces
        match = re.search(r'\{.*\}', gpt_response)
        if match:
            response_text = match.group(0)
            try:
                dict = ast.literal_eval(response_text)
            except ValueError:
                print("Error: The extracted text is not a valid dictionary.")
        else:
            print("No dictionary-like text found in the response.")
            
        return dict

    def create_cpdcf_df(self, input_pdf_path):
        field_names = self.extract_readable_form_field_names(input_pdf_path)
        keys_to_extract = ['3. Text.1', '3. Text.2', '3. CheckBox.1a','3. CheckBox.1cl','Text1.0']
        extracted_values = [field_names[key] for key in keys_to_extract if key in field_names]
        input_text = ', '.join([f'{pair[0]}: {pair[1]}' for pair in extracted_values])
        annotation = self.generate_annotation(input_text)
        return input_text, annotation

    def create_mbr_df(self, input_pdf_path):
        field_names = self.extract_readable_form_field_names(input_pdf_path)
        keys_to_extract = ['Text16', 'Text15', 'Text17','Text63','Text64','Text65','Text71','Text79','Text87','Text100','Text106']
        extracted_values = [field_names[key] for key in keys_to_extract if key in field_names]
        input_text = ', '.join([f'{pair[0]}: {pair[1]}' for pair in extracted_values])
        annotation = self.generate_annotation(input_text)
        return input_text, annotation

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[0]
        word_labels = item[1]["entities"]

        # Tokenize the text and align the labels with tokens
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_offsets_mapping=True)
        labels = [0] * self.max_len  # Initialize labels to 0 (e.g., 'O')

        last_word_idx = -1
        for word_idx, (start, end, label) in enumerate(word_labels):
            label_id = self.label_map[label]
            for token_idx, (token_start, token_end) in enumerate(encodings.offset_mapping):
                if start <= token_start and end >= token_end:  # Check if the token is part of the word
                    labels[token_idx] = label_id

        encodings.pop('offset_mapping')  # We don't need offset_mapping in the model

        # Convert input_ids and labels to PyTorch tensors
        input_ids = torch.tensor(encodings.input_ids, dtype=torch.long).flatten()
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels
    
def make_predictions(model, tokenizer, input_text):

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    model.eval()  
    outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    # Map label IDs to label names
    label_map = {0: "O", 1: "PATIENT_BIRTHDATE", 2: "DOSAGE", 3: "BIRTHDATE", 4: "DIAGNOSIS", 5:'DRUG', 6:'PATIENT_NAME', 7:'PHYSICIAN_NAME', 8:'FREQUENCY', 9:'DOB', 10:'HOSPITAL_NAME', 11:'AETNA_ID', 12:'GENDER', 13:'AMOUNT_PAID'}

    predicted_label_names = [label_map[label_id.item()] for label_id in predictions[0]]

    tokenized_input = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    aggregated_results = aggregate_subwords(tokenized_input, predicted_label_names)
    # aligned_predictions = [(word, label) for word, label in zip(tokenized_input, predicted_label_names) if word not in tokenizer.all_special_tokens]
    
    combined_results = {}

    # Temporary variables for current key and value
    current_key = None
    current_value = ""

    for token, label in aggregated_results:
        # Skip if it's a special character or non-entity
        if token in ["[CLS]", "[SEP]", "'", ",", ".", ":", ";", "!", "?"] or label == 'O':
            continue

        if label == current_key:
            # Continue building the current value
            current_value += " " + token
        else:
            if current_key is not None:
                # Save the previous key-value pair
                combined_results[current_key] = current_value.strip()

            # Start a new key-value pair
            current_key = label
            current_value = token

    # Save the last key-value pair
    if current_key is not None:
        combined_results[current_key] = current_value.strip()

    json_data = json.dumps(combined_results)

    return json_data

def aggregate_subwords(tokenized_input, predictions):
    aggregated_predictions = []
    current_word = ""
    current_label = None

    for token, label in zip(tokenized_input, predictions):
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                aggregated_predictions.append((current_word, current_label))
            current_word = token
            current_label = label

    # Add the last word
    if current_word:
        aggregated_predictions.append((current_word, current_label))

    return aggregated_predictions

def summarise_results(model, tokenizer, data_list):
    structured_summary = {
        "Diagnosis": [],
        "Prescriptions": []
    }
    
    for line in data_list:
        if 'Diagnosis:' in line:
            # Extract diagnosis
            diagnosis = line.split(": ", 1)[1]
            structured_summary["Diagnosis"].append(diagnosis)
        elif not bool(re.match(r'\d{2}/\d{2}/\d{4}', line)):
            # Process prescriptions
            inputs = tokenizer(line, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            labels = [model.config.id2label[prediction.item()] for prediction in predictions[0]]

            # Initialize variables for each part of prescription
            drug, dosage, frequency = "", "", ""
            for token, label in zip(tokens, labels):
                if label in ["B-Drug", "I-Drug"]:
                    drug += token.replace("##|▁", "") + ""
                elif label in ["B-Strength", "I-Strength"]:
                    dosage += token.replace("##|▁", "") + ""
                elif label in ["B-Frequency", "I-Frequency"]:
                    frequency += token.replace("##|▁", "") + ""

            # Clean up and add to structured summary
            drug, dosage, frequency = drug.strip(), dosage.strip(), frequency.strip()
            if drug or dosage or frequency:
                structured_summary["Prescriptions"].append({
                    "Drug": drug,
                    "Dosage": dosage,
                    "Frequency": frequency
                })

    return structured_summary