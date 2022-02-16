from transformers import BartTokenizer, BartForConditionalGeneration

def ab_summary(text: str):
    print('working')
    checkpoint = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    input_ids = tokenizer.batch_encode_plus(
        [text],
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    )
    summary_ids = model.generate(
        input_ids['input_ids']
    )
    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

def ex_summary(text: str):
    pass