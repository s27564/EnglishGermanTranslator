import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.title('Aplikacja do przetwarzania tekstu')

st.header('Przetwarzanie języka naturalnego')

options = [
        "Tłumaczenie tekstu (eng->de)",
        "Wydźwięk emocjonalny tekstu (eng)",
    ]
option = st.selectbox(
    "Opcje",
    options,
)

if option == options[0]:
    st.title("Tłumaczenie angielski > niemiecki")
    st.image(["Flag_of_the_United_Kingdom_(3-5).svg.png","Germany-Flag.jpg"],width = 100)
    text = st.text_area(label="Wpisz tekst po angielsku:", key=12)
    if text:
        with st.spinner(text='Tłumaczenie...'):
            try:
                tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

                inputs = tokenizer(text, return_tensors="pt") 
                generated_ids = model.generate(**inputs, num_beams=4, max_length=100, early_stopping=True)
    
                answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
                st.write("**Tłumaczenie:**")
                st.success(answer)
            except Exception as e:
                st.error(f"Wystąpił błąd podczas tłumaczenia: {e}")
    
     
elif option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        try:
            classifier = pipeline("sentiment-analysis")
            answer = classifier(text)
            st.write(answer)
        except Exception as e:
            st.error(f"Wystąpił błąd podczas analizy: {e}")

st.write("s27564")
