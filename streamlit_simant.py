from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from tqdm import tqdm
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range
import os
import getpass
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import pandas as pd
from langchain_qdrant import QdrantVectorStore

# Настройка страницы
st.set_page_config(
    page_title="🎬 Рекомендации сериалов",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Система рекомендации сериалов")
st.markdown("Найдите идеальный сериал по вашему запросу с помощью AI!")

# Разделитель
st.divider()

# =================================================================================================================================

# Список жанров
GENRES = [
    'Drama', 'Action&Adventure', 'Action', 'Comedy',
    'Sci-Fi&Fantasy', 'Crime', 'Animation',
    'Mystery',
    'War&Politics',
    'Soap',
    'Anime',
    'Science-Fiction',
    'Family',
    'Western',
    'Kids',
    'Fantasy',
    'Reality',
    'Documentary',
    'Romance',
    'Talk',
    'Nature',
    'Horror',
    'History',
    'Thriller',
    'Sports',
    'News'
]

# Инициализация моделей и клиента (кешируем для производительности)
@st.cache_resource
def initialize_components():
    """Инициализация всех необходимых компонентов"""
    
    # Инициализация модели эмбеддингов
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 128}
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Инициализация клиента Qdrant
    client = QdrantClient(path='qdrant_db')
    
    # Инициализация векторного хранилища
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="film_collection",
        embedding=embeddings_model
    )
    
    return vector_store, client

    
def search_movies(vector_store, query, k=7, min_rating=None, selected_genres=None):
    """Поиск фильмов по запросу пользователя с фильтрами"""
    # Создаем условия фильтра
    filter_conditions = []
    
    # Фильтр по рейтингу
    if min_rating is not None:
        filter_conditions.append(
            FieldCondition(
                key="metadata.rating",
                range=Range(gte=min_rating)
            )
        )
    
    # Фильтр по жанрам
    if selected_genres:
        filter_conditions.append(
            FieldCondition(
                key="metadata.genres",
                match=MatchAny(any=selected_genres)
            )
        )
    
    # Создаем общий фильтр если есть условия
    filter_condition = None
    if filter_conditions:
        filter_condition = Filter(must=filter_conditions)
    
    results_with_scores = vector_store.similarity_search_with_score(
        query,
        k=k,
        filter=filter_condition
    )

    # Заголовок результатов
    st.subheader(f"🎯 Найдено сериалов: {len(results_with_scores)}")
    
    # Показываем примененные фильтры
    filter_info = []
    if min_rating is not None:
        filter_info.append(f"рейтинг от {min_rating}+")
    if selected_genres:
        filter_info.append(f"жанры: {', '.join(selected_genres)}")
    
    if filter_info:
        st.caption(f"📏 Фильтры: {'; '.join(filter_info)}")
    
    if len(results_with_scores) == 0:
        st.warning("😔 По вашему запросу ничего не найдено. Попробуйте изменить параметры поиска.")
        return
    
    for i, (doc, score) in enumerate(results_with_scores):
        # Создаем контейнер для каждого результата
        with st.container():
            # Изменяем колонки: информация слева, постер справа
            col_info, col_poster = st.columns([2, 1])
            
            with col_info:
                # Блок с основной информацией
                st.markdown(f"### 🎬 {doc.metadata.get('movie name', 'Не указано')}")
                
                # Рейтинг с цветом
                rating = doc.metadata.get('rating', 'Не указано')
                if rating != 'Не указано':
                    try:
                        rating_val = float(rating)
                        if rating_val >= 8.0:
                            rating_color = "🟢"
                        elif rating_val >= 6.0:
                            rating_color = "🟡"
                        else:
                            rating_color = "🔴"
                        st.markdown(f"{rating_color} **Рейтинг:** {rating}")
                    except:
                        st.markdown(f"⭐ **Рейтинг:** {rating}")
                else:
                    st.markdown("⭐ **Рейтинг:** Не указано")
                
                st.markdown(f"📅 **Год выпуска:** {doc.metadata.get('year', 'Не указано')}")
                st.markdown(f"🎭 **Жанры:** {', '.join(doc.metadata.get('genres', []))}")
                
                # Score с визуальным индикатором
                score_percent = max(0, min(100, int(score * 100)))
                st.markdown(f"📊 **Схожесть:** {score:.4f}")
                st.progress(score_percent / 100, text=f"Релевантность: {score_percent}%")
                
                # Блок с дополнительной информацией
                st.markdown("#### 👥 Создатели")
                directors = doc.metadata.get('director', [])
                if directors:
                    st.markdown(f"**Режиссер:** {', '.join(directors)}")
                
                actors = doc.metadata.get('actors', 'Не указано')
                if actors != 'Не указано':
                    # Обрезаем длинный список актеров
                    if len(actors) > 150:
                        actors = actors[:150] + "..."
                    st.markdown(f"**Актеры:** {actors}")
                
                # Ссылки
                st.markdown("#### 🔗 Ссылки")
                page_url = doc.metadata.get('page_url', '')
                image_url = doc.metadata.get('image_url', '')
                
                if page_url and page_url != 'Не указано':
                    st.markdown(f"🌐 [Страница сериала]({page_url})")
            
            with col_poster:
                # Отображаем постер справа
                image_url = doc.metadata.get('image_url', '')
                movie_name = doc.metadata.get('movie name', 'Неизвестный сериал')
                
                if image_url and image_url != 'Не указано':
                    try:
                        st.image(
                            image_url, 
                            caption=movie_name,
                            width=300,
                            use_container_width=False,
                            output_format="auto"
                        )
                    except Exception as e:
                        st.error(f"❌ Не удалось загрузить постер: {e}")
                        st.markdown(f"🖼️ [Ссылка на постер]({image_url})")
                else:
                    st.info("📸 Постер не доступен")
                    
                    # Альтернатива - можно показать placeholder
                    st.markdown(
                        """
                        <div style='background: #f0f2f6; padding: 40px; text-align: center; border-radius: 10px;'>
                            <span style='font-size: 48px;'>🎬</span><br>
                            <span>Постер не найден</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Разделитель между результатами (кроме последнего)
            if i < len(results_with_scores) - 1:
                st.divider()

# ======================================================================================================================================

# Основной интерфейс
def main():
    # Инициализация компонентов
    try:
        vector_store, client = initialize_components()
        st.success("✅ Система рекомендаций загружена успешно!")
        
        # Сайдбар для настроек
        with st.sidebar:
            st.header("⚙️ Настройки поиска")
            
            # Количество рекомендаций
            k = st.slider(
                "Количество рекомендаций:",
                min_value=1,
                max_value=20,
                value=7,
                help="Выберите сколько сериалов показывать в результатах"
            )
            
            # Фильтры
            st.markdown("---")
            st.subheader("🎯 Фильтры")
            
            # Фильтр по рейтингу
            use_rating_filter = st.checkbox(
                "Фильтровать по рейтингу", 
                value=False,
                help="Показывать только сериалы с указанным минимальным рейтингом"
            )
            
            min_rating = None
            if use_rating_filter:
                min_rating = st.slider(
                    "Минимальный рейтинг:",
                    min_value=0.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.1,
                    help="Показывать сериалы с рейтингом не ниже указанного"
                )
                st.caption(f"🎯 Будут показаны сериалы с рейтингом {min_rating}+")
            
            # Фильтр по жанрам
            st.markdown("---")
            use_genre_filter = st.checkbox(
                "Фильтровать по жанрам",
                value=False,
                help="Показывать только сериалы выбранных жанров"
            )
            
            selected_genres = None
            if use_genre_filter:
                selected_genres = st.multiselect(
                    "Выберите жанры:",
                    options=GENRES,
                    default=[],
                    help="Можно выбрать несколько жанров"
                )
                if selected_genres:
                    st.caption(f"🎭 Будут показаны сериалы жанров: {', '.join(selected_genres)}")
                else:
                    st.caption("ℹ️ Выберите хотя бы один жанр для фильтрации")
            
            st.markdown("---")
            st.markdown("### 💡 Советы по поиску:")
            st.markdown("• Будьте конкретны в описании")
            st.markdown("• Указывайте жанры, настроение")
            st.markdown("• Можно искать по актерам или режиссерам")

        # Поле для ввода запроса пользователем
        query = st.text_input(
            "Введите ваш запрос для поиска сериалов:",
            placeholder="Например: комедийные сериалы, научная фантастика, драма про дружбу...",
            value=""
        )

        # Кнопка поиска
        if st.button("🎯 Найти сериалы", use_container_width=True):
            if query:
                with st.spinner(f"🔍 Ищем {k} подходящих сериалов..."):
                    search_movies(vector_store, query, k, min_rating, selected_genres)
            else:
                st.warning("⚠️ Пожалуйста, введите запрос для поиска")

        # Информация о текущих настройках
        filter_info = []
        if min_rating is not None:
            filter_info.append(f"рейтинг от {min_rating}+")
        if selected_genres:
            filter_info.append(f"жанры: {', '.join(selected_genres)}")
        
        filter_text = f" ({'; '.join(filter_info)})" if filter_info else ""
        st.info(f"📊 Будет показано: **{k} рекомендаций{filter_text}**")

    except Exception as e:
        st.error(f"❌ Ошибка загрузки системы: {e}")
        return

if __name__ == "__main__":
    main()