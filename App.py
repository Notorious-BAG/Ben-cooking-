import math
import re
import gkeepapi
import requests
import streamlit as st

# =========================================================
# 1. Recommendation Engine & Rule-Based Filters
# =========================================================

STOP_WORDS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon", "tsp",
    "pound", "pounds", "lb", "lbs", "ounce", "ounces", "oz", "gram", "grams", "g",
    "kg", "clove", "cloves", "pinch", "bunch", "sliced", "diced", "chopped",
    "minced", "fresh", "freshly", "ground", "large", "small", "medium", "water",
    "salt", "black pepper", "taste", "to", "and", "or", "for"
}

NON_VEGETARIAN_STEMS = [
    "chicken", "beef", "pork", "bacon", "turkey", "lamb", "duck", "veal", "ham",
    "prosciutto", "pancetta", "sausage", "pepperoni", "chorizo", "steak",
    "salmon", "tuna", "cod", "shrimp", "prawn", "crab", "lobster", "clam",
    "mussel", "oyster", "anchovy", "anchovies", "sardine", "tilapia", "halibut",
    "gelatin", "lard", "tallow", "bone broth", "chicken broth", "beef broth",
    "chicken stock", "beef stock", "fish sauce", "worcestershire"
]

def check_vegetarian(recipe: dict):
    """Audits ingredients against meat and byproduct stems."""
    text = " ".join(recipe.get("ingredients", [])).lower()
    violations = []
    for stem in NON_VEGETARIAN_STEMS:
        if re.search(rf"\b{stem}\b", text):
            violations.append(stem)
    return len(violations) == 0, list(set(violations))

def calculate_complexity(recipe: dict):
    """Calculates normalized complexity score K in [0, 1]."""
    minutes = recipe.get("readyInMinutes", 30)
    ingredients_count = len(recipe.get("ingredients", []))
    steps_count = len(recipe.get("instructions", []))

    time_score = min(1.0, minutes / 90.0)
    ing_score = min(1.0, ingredients_count / 20.0)
    step_score = min(1.0, steps_count / 15.0)

    k = (0.45 * time_score) + (0.35 * ing_score) + (0.20 * step_score)
    label = "Quick & Simple" if k < 0.35 else ("Moderate" if k <= 0.65 else "High Effort")
    return round(k, 3), label

def extract_features(recipe: dict):
    """Extracts ingredient tokens and bigrams as a weighted bag-of-words."""
    features = {}
    for raw_ing in recipe.get("ingredients", []):
        cleaned = re.sub(r"[\d\/\.\(\),]", "", raw_ing.lower()).split()
        tokens = [w for w in cleaned if len(w) > 2 and w not in STOP_WORDS]
        for i, token in enumerate(tokens):
            features[token] = features.get(token, 0.0) + 1.0
            if i < len(tokens) - 1:
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                features[bigram] = features.get(bigram, 0.0) + 1.5
    return features

def calculate_match_score(recipe: dict, taste_profile: dict, save_count: int, avg_complexity: float, vegetarian_only: bool):
    """Calculates overall recommendation percentage combining Cosine Similarity, Complexity, and Diet."""
    is_veg, violations = check_vegetarian(recipe)
    if vegetarian_only and not is_veg:
        return 0, False, f"Blocked: Contains {', '.join(violations[:2])}", []

    if save_count < 2 or not taste_profile:
        return None, is_veg, "Cold start (Save 2+ recipes to train taste profile)", []

    candidate_features = extract_features(recipe)

    # Cosine similarity
    dot_product = sum(taste_profile[k] * w for k, w in candidate_features.items() if k in taste_profile)
    profile_norm = math.sqrt(sum(w * w for w in taste_profile.values()))
    candidate_norm = math.sqrt(sum(w * w for w in candidate_features.values()))

    if profile_norm == 0 or candidate_norm == 0:
        return 0, is_veg, "Neutral match", []

    cosine_sim = dot_product / (profile_norm * candidate_norm)

    # Gaussian complexity multiplier
    k_recipe, _ = calculate_complexity(recipe)
    delta_comp = abs(k_recipe - avg_complexity)
    complexity_mult = 0.65 + (0.35 * math.exp(- (delta_comp ** 2) / (2 * (0.25 ** 2))))

    score = min(100, int(((cosine_sim ** 0.6) * 100) * complexity_mult))
    matched_features = sorted(
        [k for k in candidate_features if k in taste_profile],
        key=lambda k: taste_profile[k] * candidate_features[k],
        reverse=True
    )[:3]

    return score, is_veg, "", matched_features

def update_profile(recipe: dict):
    """Updates the user taste vector and average complexity in session storage."""
    features = extract_features(recipe)
    for k, w in features.items():
        st.session_state.taste_profile[k] = st.session_state.taste_profile.get(k, 0.0) + w

    k_recipe, _ = calculate_complexity(recipe)
    count = st.session_state.save_count
    st.session_state.avg_complexity = ((st.session_state.avg_complexity * count) + k_recipe) / (count + 1)
    st.session_state.save_count += 1

# =========================================================
# 2. Google Keep Exporter
# =========================================================

@st.cache_resource
def get_keep_client(email: str, app_password: str):
    keep = gkeepapi.Keep()
    keep.authenticate(email, app_password)
    return keep

def export_recipe_to_keep(keep, recipe: dict):
    """Creates a checklist note with ingredients and instructions in Keep."""
    items = [(f"{item}", False) for item in recipe.get("ingredients", [])]
    note = keep.createList(f"🍲 {recipe['title']}", items)

    body = f"Source: {recipe.get('url', 'N/A')}\nTime: {recipe.get('readyInMinutes', 30)} min\n\nInstructions:\n"
    for idx, step in enumerate(recipe.get("instructions", []), 1):
        body += f"{idx}. {step}\n"
    note.text = body

    label = keep.findLabel("Recipes") or keep.createLabel("Recipes")
    note.labels.add(label)
    note.color = gkeepapi.node.ColorValue.Yellow
    keep.sync()

# =========================================================
# 3. Recipe Provider (Spoonacular)
# =========================================================

def fetch_recipes(api_key: str, tags: str = "", count: int = 5):
    endpoint = "https://api.spoonacular.com/recipes/random"
    params = {"apiKey": api_key, "number": count, "tags": tags}
    resp = requests.get(endpoint, params=params, timeout=10)
    if resp.status_code != 200:
        st.error(f"Spoonacular API Error: {resp.json().get('message', 'Failed to fetch recipes')}")
        return []

    results = []
    for r in resp.json().get("recipes", []):
        ingredients = [ing.get("original") for ing in r.get("extendedIngredients", [])]
        instructions = [
            s.get("step")
            for section in r.get("analyzedInstructions", [])
            for s in section.get("steps", [])
        ]
        results.append({
            "title": r.get("title"),
            "image": r.get("image", ""),
            "readyInMinutes": r.get("readyInMinutes", 30),
            "servings": r.get("servings", 2),
            "url": r.get("sourceUrl", ""),
            "ingredients": ingredients,
            "instructions": instructions
        })
    return results

# =========================================================
# 4. Streamlit Application Interface
# =========================================================

st.set_page_config(page_title="Recipe Matcher & Keep Exporter", page_icon="🍳")

# Initialize persistent memory in session state
if "recipes" not in st.session_state:
    st.session_state.recipes = []
if "index" not in st.session_state:
    st.session_state.index = 0
if "taste_profile" not in st.session_state:
    st.session_state.taste_profile = {}
if "save_count" not in st.session_state:
    st.session_state.save_count = 0
if "avg_complexity" not in st.session_state:
    st.session_state.avg_complexity = 0.5

with st.sidebar:
    st.header("Settings & Credentials")
    user_email = st.text_input("Google Email", placeholder="user@gmail.com")
    app_password = st.text_input("Google App Password", type="password", help="Generate via Google Account > Security > 2-Step Verification > App passwords")
    spoonacular_key = st.text_input("Spoonacular API Key", type="password")
    
    st.divider()
    vegetarian_only = st.checkbox("Enforce Strict Vegetarian", value=False)
    api_diet_tag = "vegetarian" if vegetarian_only else ""

    if st.button("Reset Taste Profile"):
        st.session_state.taste_profile = {}
        st.session_state.save_count = 0
        st.session_state.avg_complexity = 0.5
        st.success("Preferences reset.")

if not (user_email and app_password and spoonacular_key):
    st.info("👈 Enter your Google App Password and Spoonacular API Key in the sidebar to start.")
    st.stop()

# Fetch batch if empty
if not st.session_state.recipes or st.session_state.index >= len(st.session_state.recipes):
    with st.spinner("Finding fresh recipes..."):
        st.session_state.recipes = fetch_recipes(spoonacular_key, tags=api_diet_tag)
        st.session_state.index = 0

if st.session_state.recipes:
    recipe = st.session_state.recipes[st.session_state.index]

    # Evaluate against the recommendation engine
    score, is_veg, block_reason, matched_terms = calculate_match_score(
        recipe,
        st.session_state.taste_profile,
        st.session_state.save_count,
        st.session_state.avg_complexity,
        vegetarian_only
    )
    k_val, comp_label = calculate_complexity(recipe)

    # Badges Header
    col_b1, col_b2, col_b3 = st.columns(3)
    if score is not None:
        col_b1.metric("Recommendation", f"{score}% Match")
    else:
        col_b1.caption("Rating: Learning your tastes")

    col_b2.metric("Dietary Status", "🌱 Vegetarian" if is_veg else "🥩 Non-Veg")
    col_b3.metric("Complexity", comp_label)

    if block_reason and vegetarian_only:
        st.error(block_reason)

    if matched_terms:
        st.caption(f"Matches your affinity for: **{', '.join(matched_terms)}**")

    # Recipe Card
    st.subheader(recipe["title"])
    if recipe["image"]:
        st.image(recipe["image"], use_container_width=True)

    col_m1, col_m2 = st.columns(2)
    col_m1.write(f"⏱️ **Time:** {recipe['readyInMinutes']} mins")
    col_m2.write(f"🍽️ **Servings:** {recipe['servings']}")

    with st.expander("View Ingredients & Steps"):
        st.markdown("**Ingredients Checklist:**")
        for item in recipe["ingredients"]:
            st.markdown(f"- [ ] {item}")
        st.markdown("**Instructions:**")
        for idx, step in enumerate(recipe["instructions"], 1):
            st.markdown(f"{idx}. {step}")

    # Action Controls
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("❌ Pass / Next", use_container_width=True):
            st.session_state.index += 1
            st.rerun()

    with btn_col2:
        can_save = not (vegetarian_only and not is_veg)
        if st.button("❤️ Save to Google Keep", use_container_width=True, disabled=not can_save):
            try:
                keep = get_keep_client(user_email, app_password)
                export_recipe_to_keep(keep, recipe)
                update_profile(recipe)
                st.success(f"Saved '{recipe['title']}' to Keep!")
                st.session_state.index += 1
                st.rerun()
            except Exception as e:
                st.error(f"Keep sync failed: {e}")
