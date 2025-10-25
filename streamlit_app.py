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
        st.spinner()
        with st.spinner(text='Tłumaczenie...'):
            try:
                tokenizer = AutoTokenizer.from_pretrained("DunnBC22/mbart-large-50-English_German_Translation")
                model = AutoModelForSeq2SeqLM.from_pretrained("DunnBC22/mbart-large-50-English_German_Translation")

                tokenizer.src_lang = "en_XX"
                tokenizer.tgt_lang = "de_DE"

                inputs = tokenizer(text, return_tensors="pt") 
                generated_ids = model.generate(**inputs, num_beams=4, max_length=100, early_stopping=True)
    
                answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
            except Exception as e:
                st.error(f"Wystąpił błąd podczas tłumaczenia: {e}")
    
        st.write("**Tłumaczenie:**")
        st.success(answer)
     
elif option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)

st.write("s27564")

#st.subheader('Zadanie do wykonania')
#st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
#st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
#st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
#st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
#st.write('🐞 Na końcu umieść swój numer indeksu')
#st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
#st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')
