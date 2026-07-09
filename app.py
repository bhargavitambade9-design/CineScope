import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineScope | Movie Recommender", page_icon="🎬", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "cleaned_data1.csv")


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Movie data not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def normalize_title(title: str) -> str:
    return "".join(ch for ch in str(title).lower() if ch.isalnum())


@st.cache_resource(show_spinner=False)
def build_similarity_matrix(movies_df):
    tfidf = TfidfVectorizer(stop_words="english")
    vector = tfidf.fit_transform(movies_df["tags"]).toarray()
    return cosine_similarity(vector)


def get_index_from_name(movie_name: str, movies_df):
    clean_name = normalize_title(movie_name)
    for idx, title in enumerate(movies_df["title"]):
        if normalize_title(title) == clean_name:
            return idx
    return None


def get_recommendations(movie_name: str, movies_df, similarity_matrix, top_n: int = 5):
    index = get_index_from_name(movie_name, movies_df)
    if index is None:
        return []

    similarity_scores = sorted(enumerate(similarity_matrix[index]), key=lambda item: item[1], reverse=True)

    recommendations = []
    for movie_idx, score in similarity_scores[1 : top_n + 1]:
        recommendations.append(
            {
                "title": movies_df.loc[movie_idx, "title"],
                "score": round(float(score), 3),
            }
        )
    return recommendations


poster_map = {
    "Avatar": "https://image.tmdb.org/t/p/w500/kqjL17yufvn9OVLyXYpvtyrFfak.jpg",
    "Pirates of the Caribbean: At World's End": "https://image.tmdb.org/t/p/w500/yXgMh3W6y9w5V4ELh4kcQtxFQdi.jpg",
    "Spectre": "https://image.tmdb.org/t/p/w500/pFlaoHTZeyNkG83vxsAJiGzfSsa.jpg",
    "The Dark Knight Rises": "https://image.tmdb.org/t/p/w500/hdl5sGxF1kjTesjOFH1kFl5RdV.jpg",
    "John Carter": "https://image.tmdb.org/t/p/w500/6ufzvSwdxD9Z9tUOQn3aLIOcJXP.jpg",
    "Inception": "https://image.tmdb.org/t/p/w500/qmDpIHrmpJINaRKAfWQfftjCdyi.jpg",
    "Titanic": "https://image.tmdb.org/t/p/w500/kHXEpyfl6zqn8a6YuozZUujufXf.jpg",
}


def get_poster_url(title: str) -> str:
    return poster_map.get(title, f"https://via.placeholder.com/300x450/121f34/ffffff?text={title.replace(' ', '+')}")


def featured_movies(movies_df, count: int = 4):
    featured_titles = [
        "Avatar",
        "Pirates of the Caribbean: At World's End",
        "Inception",
        "The Dark Knight Rises",
    ]
    return [title for title in featured_titles if title in movies_df["title"].values][:count]


movies_df = load_data()
similarity_matrix = build_similarity_matrix(movies_df)
featured = featured_movies(movies_df)

if "watch_history" not in st.session_state:
    st.session_state.watch_history = []

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None


def select_movie(movie_title: str):
    st.session_state.selected_movie = movie_title
    if movie_title and movie_title not in st.session_state.watch_history:
        st.session_state.watch_history.append(movie_title)

page_style = """
<style>
body {
    background: linear-gradient(180deg, #08111f 0%, #0d2844 100%);
    color: #f5f8ff;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
section {
    padding: 0.5rem 0rem;
}
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 18px;
    padding: 12px 18px;
    border-radius: 24px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.top-bar__brand {
    flex: 1;
    text-align: center;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    color: #ffffff;
    text-transform: uppercase;
}
.top-bar__login {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.06);
    color: #ffffff;
    font-weight: 700;
    transition: transform 0.18s ease, background 0.18s ease;
}
.top-bar__login:hover {
    transform: translateY(-1px);
    background: rgba(255,255,255,0.12);
}
.header-hero {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 32px;
    padding: 32px;
    box-shadow: 0 40px 110px rgba(0, 0, 0, 0.35);
    margin-bottom: 24px;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: 999px;
    background: rgba(255, 90, 95, 0.18);
    color: #ffb2ae;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 18px;
}
.header-title {
    margin: 0;
    font-size: 3.3rem;
    line-height: 1.02;
    letter-spacing: 0.02em;
    color: #ffffff;
}
.header-title em {
    color: #ff8c76;
    font-style: normal;
}
.header-subtitle {
    margin: 18px 0 0 0;
    color: #b8c6e5;
    font-size: 1.05rem;
    line-height: 1.8;
}
.header-stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 24px;
}
.header-stat {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px 20px;
    border-radius: 20px;
    color: #e5f0ff;
}
.header-stat strong {
    display: block;
    font-size: 1.5rem;
    margin-bottom: 6px;
}
.search-panel {
    background: rgba(16, 30, 58, 0.95);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 35px 90px rgba(0, 0, 0, 0.35);
    margin-bottom: 18px;
}
.search-panel h3 {
    margin: 0 0 10px 0;
    font-size: 1.4rem;
    color: #ffffff;
}
.search-panel p {
    margin: 0 0 18px 0;
    color: #b8c6e5;
    line-height: 1.6;
}
.stButton>button {
    background: linear-gradient(135deg, #ff5a5f 0%, #ff8f6e 100%);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 0.95rem 1.8rem;
    font-weight: 700;
    box-shadow: 0 16px 28px rgba(255, 90, 95, 0.24);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 20px 35px rgba(255, 90, 95, 0.28);
}
.stTextInput>div>div>input {
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.08);
    color: white;
}
.css-1vqgbsn, .st-c7 {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 24px !important;
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
}
</style>
"""

st.markdown(page_style, unsafe_allow_html=True)

st.markdown(
    """
    <div class='top-bar'>
        <div style='width:110px;'></div>
        <div class='top-bar__brand'>CineScope</div>
        <a class='top-bar__login' href='#'>Login</a>
    </div>
    <div class='header-hero'>
        <div class='header-badge'>
            <span style='display:inline-block; width:10px; height:10px; border-radius:999px; background:#ff8c76; box-shadow:0 0 18px rgba(255,140,118,0.45);'></span>
            <span>Premium Recommendation Lab</span>
        </div>
        <div style='display:flex; flex-wrap:wrap; justify-content:space-between; gap:18px;'>
            <div style='max-width:760px;'>
                <h1 class='header-title'>Find your next<br><em>cinema thrill</em> with CineScope.</h1>
                <p class='header-subtitle'>Instant movie matches based on your watched titles, poster-rich recommendations, and a deluxe homepage vibe designed for film lovers.</p>
            </div>
            <div class='header-stats'>
                <div class='header-stat'><strong>5M+</strong> curated suggestions</div>
                <div class='header-stat'><strong>24/7</strong> recommendation engine</div>
                <div class='header-stat'><strong>90%</strong> user satisfaction</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            """
            <div style='padding: 32px; border-radius: 30px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);'>
                <p style='color:#f7c46c; font-weight:700; letter-spacing:0.14em; margin-bottom:0.75rem;'>CINESCOPE RECOMMENDER</p>
                <h1 style='font-size:3rem; line-height:1.05; margin:0;'>Smart picks for your next movie night.</h1>
                <p style='color:#cdd7ee; font-size:1.05rem; margin-top:1.2rem;'>Enter a title, explore curated matches, and enjoy cinematic suggestions with premium poster styling.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.image(
            "https://images.unsplash.com/photo-1524985069026-dd778a71c7b4?auto=format&fit=crop&w=900&q=80",
            caption="Browse trending movies",
            width=420,
        )

with st.container():
    announcement, search = st.columns([1, 2])
    with announcement:
        st.markdown(
            """
            <div style='background:#11263d; padding:28px; border-radius:28px; border:1px solid rgba(255,255,255,0.08);'>
                <p style='color:#ffb547; margin-bottom:10px; font-size:0.95rem;'>WHAT'S NEW</p>
                <h3 style='margin-top:0; margin-bottom:0.75rem;'>Unlock fresh picks today</h3>
                <p style='color:#c8d7f5; margin-bottom:1rem;'>Discover new release posters, improved suggestion accuracy, and a clean homepage style inspired by premium entertainment apps.</p>
                <ul style='color:#a9bfe8; margin-left:18px; padding-left:0;'>
                    <li>Recommended movies based on watched titles</li>
                    <li>Real poster previews for featured titles</li>
                    <li>Smooth, dark UI with accent colors</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with search:
        st.markdown(
            """
            <div class='search-panel'>
                <h3>Search the movie universe</h3>
                <p>Type a title and let CineScope find your next binge-worthy pick with rich recommendations and cinematic style.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form(key="search_form"):
            search_box = st.text_input("Search movies", placeholder="Try: Avatar, Inception, Titanic")
            search_button = st.form_submit_button("Reveal Suggestions")

        if st.session_state.selected_movie:
            search_box = st.session_state.selected_movie
            recommendations = get_recommendations(search_box, movies_df, similarity_matrix)
            title_label = f"Because you watched <span style='color:#ff7f71;'>{search_box}</span>"
        elif search_button and search_box.strip():
            recommendations = get_recommendations(search_box, movies_df, similarity_matrix)
            title_label = f"Because you watched <span style='color:#ff7f71;'>{search_box}</span>"
            if recommendations and search_box not in st.session_state.watch_history:
                st.session_state.watch_history.append(search_box)
        else:
            recommendations = []
            title_label = ""

        if recommendations:
            st.markdown(
                f"<div style='padding:24px; border-radius:24px; background:rgba(18,37,59,0.9); border:1px solid rgba(255,255,255,0.06); margin-bottom:16px;'>"
                f"<h3 style='margin:0;'>{title_label}</h3>"
                f"</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            for idx, movie in enumerate(recommendations):
                with cols[idx % 2]:
                    poster_url = get_poster_url(movie["title"])
                    st.markdown(
                        f"<div style='background:#0f233e; padding:18px; border-radius:22px; margin-bottom:18px; border:1px solid rgba(255,255,255,0.08);'>"
                        f"<img src='{poster_url}' style='width:100%; border-radius:18px; margin-bottom:12px;' />"
                        f"<h4 style='margin:0 0 6px 0; color:#ffffff;'>{movie['title']}</h4>"
                        f"<p style='margin:0 0 12px 0; color:#a9c6ff;'>Similarity: {movie['score']}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Explore similar",
                        key=f"recommend_{idx}_{movie['title']}",
                        on_click=select_movie,
                        args=(movie["title"],),
                    )
        elif search_button and search_box.strip():
            st.error("Movie not found. Please try another title.")
        elif not st.session_state.selected_movie and search_button:
            st.info("Please enter a movie name.")

with st.container():
    st.markdown("<h2 style='margin-bottom:18px;'>New releases</h2>", unsafe_allow_html=True)
    cols = st.columns(len(featured))
    for index, title in enumerate(featured):
        with cols[index]:
            st.markdown(
                f"<div style='border-radius:24px; overflow:hidden; box-shadow:0 25px 70px rgba(0,0,0,0.25);'>"
                f"<img src='{get_poster_url(title)}' style='width:100%; display:block;' />"
                f"<div style='padding:14px; background:rgba(0,0,0,0.5);'>"
                f"<p style='margin:0; font-weight:700; color:#fff;'>{title}</p>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.button(
                f"View {title}",
                key=f"featured_{index}",
                on_click=select_movie,
                args=(title,),
            )

with st.container():
    st.markdown(
        """
        <div style='background:#11263d; padding:24px; border-radius:24px; border:1px solid rgba(255,255,255,0.08);'>
            <h4 style='color:#ff8a65; margin-bottom:12px;'>Suggestion box</h4>
            <p style='color:#d3e1f3; margin-bottom:10px;'>Type the name of a movie you watched and hit Get Suggestions to receive a curated list of similar titles displayed with poster cards.</p>
            <p style='color:#a8bcd8; margin:0;'>Try a popular movie title like Avatar, Inception, Titanic, or The Dark Knight Rises.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.container():
    if st.session_state.watch_history:
        st.markdown("<h2 style='margin-bottom:18px;'>Recently watched</h2>", unsafe_allow_html=True)
        for idx, watched in enumerate(reversed(st.session_state.watch_history)):
            st.markdown(
                f"<div style='background:#0f233e; padding:14px; border-radius:18px; border:1px solid rgba(255,255,255,0.08); margin-bottom:10px;'>"
                f"<p style='margin:0 0 10px 0; color:#ff9c85; font-weight:600;'>{watched}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.button(
                f"Replay {watched}",
                key=f"history_{idx}_{watched}",
                on_click=select_movie,
                args=(watched,),
            )

with st.container():
    st.markdown(
        """
        <div style='text-align:center; padding:18px; margin-top:24px; color:#8fa6d4;'>
            <p style='margin:0;'>Built with Python, Streamlit, and TMDB-style movie recommendations.</p>
            <p style='margin:0;'>Powered by your watched movie history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
