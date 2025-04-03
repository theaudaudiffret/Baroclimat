from nltk.corpus import stopwords

# Translate config
LENGTH_BATCH_TRANSLATION = 10

# Train config
TEST_SIZE = 0.3
MAX_LEN = 512
LABELS_CLIMATE = ["no_climate", "climate"]
N_EPOCHS = 10
LR = 2e-5
BATCH_SIZE = 16

# Inference config
DEFAULT_MODEL_ID = "2024_03_24_14_08_15"
DEFAULT_EMBED_MODEL_ID = "cc.fr.300"

# Storage config
STORAGE_CONTAINER_MODELS = "models"
STORAGE_CONTAINER_RAW_DATA = "raw-data"
STORAGE_CONTAINER_RESULTS = "inference"

# Climate topics
PATH_CLIMATE_TOPICS = "src/baroclimat/config/climate_topics.toml"

# Preprocessing
FRENCH_STOPWORDS = set(stopwords.words("french"))
ENGLISH_STOPWORDS = set(stopwords.words("english"))
