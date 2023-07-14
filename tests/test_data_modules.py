from adjunct.data_modules import ICD10DataModule

def test_icd_data_module():
    test_checkpoint = "bert-base-uncased"
    test_data_checkpoint = "lowem1/cms-icd10-categorical"
    _ = ICD10DataModule(
        tokenizer_checkpoint=test_checkpoint,
        data_checkpoint=test_data_checkpoint,
    )

    assert _.tokenizer.name_or_path == test_checkpoint
def test():
    test_icd_data_module()