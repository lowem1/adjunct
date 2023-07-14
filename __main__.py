from adjunct.data_modules import ICD10DataModule


mod = ICD10DataModule(tokenizer_checkpoint="bert-base-uncased", data_checkpoint="lowem1/cms-icd10-categorical")