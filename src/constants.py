PROJECT_VERSION = "1.0.0"
STATISTICS_SCHEMA_VERSION = 1

LABELS = (
    "foreign",
    "out_of_focus",
    "air/bubble",
    "fold/wrinkle",
    "pen",
)

# Filename-safe label names for output files.
LABEL_FILE_SLUGS = {
    "foreign": "foreign",
    "out_of_focus": "out_of_focus",
    "air/bubble": "air_bubble",
    "fold/wrinkle": "fold_wrinkle",
    "pen": "pen",
}

# Label colors (RGB)
LABEL_COLORS = {
    "foreign": (231, 76, 60),
    "out_of_focus": (241, 196, 15),
    "air/bubble": (52, 152, 219),
    "fold/wrinkle": (230, 126, 34),
    "pen": (127, 140, 141),
}

SUPPORTED_MODEL_NAMES = (
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_large",
)

THRESHOLD_PROFILES = ("balanced", "recall")
SUPPORTED_WSI_EXTENSIONS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")

MODEL_INPUT_PATCH_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Heatmap aggregation
# Each band adds weight to an inset region of the patch.
# Cumulative weights: 0.20 (edge), 0.50 (middle), 1.00 (center).
HEATMAP_BLEND_BANDS = (
    (0.000, 0.20),
    (0.125, 0.30),
    (0.250, 0.50),
)
