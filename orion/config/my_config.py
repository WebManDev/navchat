import torch
from attr import define


@define
class MapConfig:
    num_grid: int = 600
    num_vxl_height: int = (
        40  # should be larger than (ceiling_height_wrt_camera+camera_height)/cell_size
    )
    downsample_factor: int = 1
    cell_size: float = 0.05
    min_depth: float = 0.1  # meter
    max_depth: float = 10.0  # meter
    screen_h: int = 480
    screen_w: int = 640
    fov: int = 90
    fx: float = 1169.621094
    fy: float = 1167.105103
    cx: float = 646.295044
    cy: float = 489.927032
    ceiling_height_wrt_camera: float = 0.7
    camera_height: float = 0.88
    agent_height_tolerance: float = 0.2
    laser_scan_limit: int = 10

    wheel_radius: float = 0.18

    blind_dist: int = 30  # int(camera_height/screen_h*screen_w/cell_size)+2 # cells
    # the distance that are not visible because of the camera's blind spot in the front
    # since we assume the camera can not pan up and down.

    def update_with_habitat_config(self, habitat_config):
        """used when launched the habitat simulator and need to update the map config"""
        self.camera_height = habitat_config.SIMULATOR.AGENT_0.HEIGHT
        self.screen_h = habitat_config.SIMULATOR.RGB_SENSOR.HEIGHT
        self.screen_w = habitat_config.SIMULATOR.RGB_SENSOR.WIDTH
        self.fov = habitat_config.SIMULATOR.RGB_SENSOR.HFOV
        self.min_depth = habitat_config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        self.max_depth = habitat_config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH


#############  feature extractor config  #############
@define
class CLIPConfig:
    """
    ViT-B/16 68.3% openai
    ViT-B/32 63.2% openai
    ViT-L/14 75.5% openai
    ViT-L/14 79.2% datacomp_xl_s13b_b90k
    """

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"  # default
    clip_version: str = "ViT-B-16"  # default
    openclip_pretained: str = "openai"  # default
    height: int = 480
    width: int = 640


@define
class CLIPConfig_vitB32_openai(CLIPConfig):
    clip_version: str = "ViT-B-32"
    openclip_pretained: str = "openai"


@define
class CLIPConfig_vitL14_openai(CLIPConfig):
    clip_version: str = "ViT-L-14"
    openclip_pretained: str = "openai"


@define
class CLIPConfig_vitL14_datacomp(CLIPConfig):
    clip_version: str = "ViT-L-14"
    openclip_pretained: str = "datacomp_xl_s13b_b90k"


@define
class LsegConfig(CLIPConfig):
    device: str = torch.device("cpu")  # using cpu for compatibility
    # vision backbone: ViT-L/16, text encoder: CLIP ViT-B/32
    ckpt_path: str = "data/pretrained_ckpts/lseg_demo_e200.ckpt"
    clip_version: str = "ViT-B-32"
    openclip_pretained: str = "openai"
    mode: str = "extraction"  # "extraction" or "inference"
    # threshold:float=10 # for vlmap index, if logits too small, then is "other"


@define
class ConceptFusionConfig(CLIPConfig):
    sam_model_type: str = "vit_h"
    sam_ckpt_path: str = "data/pretrained_ckpts/sam_vit_h_4b8939.pth"
    mode: str = "extraction"  # "extraction" or "inference"
    # threshold:float=0.5


VLMAP_QUERY_LIST_BASE = [
    "other",
    "wall",
    "floor",
    "stairs",
    "door",
    "window",
    "bed",
    "couch",
    "toilet",
    "table",
    "chair",
    "sink",
    "shelf",
    "cabinet",
    "fridge",
]
VLMAP_QUERY_LIST_COMMON = VLMAP_QUERY_LIST_BASE + [
    "freezer",
    "carpet",
]


######## Object Detector Config ########


@define
class GroundingDINOConfig:
    config_file = "third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "data/pretrained_ckpts/groundingdino_swint_ogc.pth"
    sam_checkpoint = "data/pretrained_ckpts/sam_vit_h_4b8939.pth"
    box_threshold = 0.4
    text_threshold = 0.25
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )


######### Fontier-Based Exploration  Config #########
@define
class FBEConfig:
    # get the frontier piece
    size_thres: int = 7  # size threshold for the frontier piece
    var_thres: float = 3  # variance threshold for the frontier piece
    size_large_thres: int = 11  # size threshold for the large frontier piece
    dist_small_thres: int = 10  # distance threshold for the small frontier piece
    dist_large_thres: int = 15  # distance threshold for the large frontier piece
    fast_explore_forwardcount: int = 5
    # fast explore go directly to the ceter of the frontier piece
    # slow explore go to the frontier center, and then turn left and right 180 degree to look around.


###### Chosen Scenes for Experiments ######
# We manually selected a few scenes for experiments, which are used in the paper.

# (scene_id, (min_floor_heigh, max_floor_height)
# Our agent can only navigate in one floor, can not go up and down the stairs.
SCENE_ID_FLOOR_SET = [
    ("4ok3usBNeis", (-1, 1)),  # many objects
    ("TEEsavR23oF", (2, 5)),  # many objects
    ("y9hTuugGdiq", (-2, 2)),  # has mirror & glass window
    ("MHPLjHsuG27", (-2, 2)),  # has mirror & glass window
    ("h1zeeAwLh9Z", (2, 5)),
    ("mL8ThkuaVTM", (-2, 2)),
    ("QaLdnwvtxbs", (-2, 2)),
    ("qyAac8rV8Zk", (-2, 2)),
    ("LT9Jq6dN3Ea", (-2, 2)),
    ("cvZr5TUy5C5", (-2, 2)),
]


#### User Simulator Config ####
WALL_OBJECTS = [
    "wall",
    "door",
    "window",
    "blinds",
    "cabinet door",
    "paneling",
    "window frame",
    "closet door",
    "board",
    "shades",
    "balustrade",
    "curtain",
]
FILTER_OBJECTS = [
    "frame",
    "door",
    "object",
    "rug",
    "tap",
    "boiler",
    "wall",
    "mat",
    "beam",
    "carpet",
    "display cabinet",
    "bucket",
    "curtain",
    "photo",
    "sheet",
    "basket",
    "bag",
    "shirt",
    "balustrade",
    "decoration",
    "pillow",
    "board",
    "handrail",
    "chandelier",
    "blanket",
    "stair",
    "floor",
    "ceiling",
    "picture",
    "painting",
    "heater",
    "banister",
    "pillar",
    "bowl",
    "towel",
    "ottoman",
    "pouffe",
    "ornament",
]
ROOMS = [
    "bedroom",
    "bathroom",
    "kitchen",
    "living room",
]  # only bedromm and bathroom can be multiple in persona

# Dictionary mapping objects to likely rooms
# Used for goToRoom functionality - helps determine which room to navigate to
OBJECT_TO_ROOMS = {
    # Bedroom objects
    "bed": ["bedroom"],
    "nightstand": ["bedroom"],
    "dresser": ["bedroom"],
    "wardrobe": ["bedroom"],
    "bedside lamp": ["bedroom"],
    "table lamp": ["bedroom"],
    "shoe": ["bedroom", "closet", "hallway"],
    "shoes": ["bedroom", "closet", "hallway"],
    "clothes": ["bedroom", "closet"],
    "pillow": ["bedroom"],
    "blanket": ["bedroom"],
    "closet": ["bedroom", "hallway"],
    "mirror": ["bedroom", "bathroom"],
    "armchair": ["bedroom", "living room"],
    "lounge chair": ["bedroom", "living room"],
    
    # Bathroom objects
    "toilet": ["bathroom"],
    "bathtub": ["bathroom"],
    "shower": ["bathroom"],
    "sink": ["bathroom", "kitchen"],
    "bathroom cabinet": ["bathroom"],
    "bathroom counter": ["bathroom"],
    "towel": ["bathroom"],
    "towel rack": ["bathroom"],
    "bathroom mirror": ["bathroom"],
    "soap": ["bathroom"],
    "toilet paper": ["bathroom"],
    
    # Kitchen objects
    "refrigerator": ["kitchen"],
    "fridge": ["kitchen"],
    "freezer": ["kitchen"],
    "oven": ["kitchen"],
    "stove": ["kitchen"],
    "microwave": ["kitchen"],
    "kitchen cabinet": ["kitchen"],
    "kitchen counter": ["kitchen"],
    "kitchen shelf": ["kitchen"],
    "cabinet": ["kitchen", "bathroom", "living room"],
    "sink": ["kitchen", "bathroom"],
    "dishwasher": ["kitchen"],
    "dish": ["kitchen"],
    "bowl": ["kitchen"],
    "cupboard": ["kitchen"],
    "pantry": ["kitchen"],
    "trashcan": ["kitchen", "bathroom"],
    "garbage can": ["kitchen"],
    "cutting board": ["kitchen"],
    "knife": ["kitchen"],
    
    # Living room objects
    "couch": ["living room"],
    "sofa": ["living room"],
    "tv": ["living room", "bedroom"],
    "led tv": ["living room", "bedroom"],
    "television": ["living room", "bedroom"],
    "coffee table": ["living room"],
    "side table": ["living room"],
    "end table": ["living room"],
    "table": ["living room", "dining room", "kitchen", "bedroom"],
    "carpet": ["living room"],
    "rug": ["living room"],
    "fireplace": ["living room"],
    "lamp": ["living room", "bedroom"],
    "bookshelf": ["living room", "bedroom", "office room"],
    "shelf": ["living room", "bedroom", "kitchen"],
    "rack": ["living room", "bedroom"],
    "speaker": ["living room"],
    "stereo": ["living room"],
    "recliner": ["living room"],
    "ottoman": ["living room"],
    "pouffe": ["living room"],
    "chair": ["living room", "bedroom", "dining room", "office room"],
    "desk": ["living room", "bedroom", "office room"],
    "computer desk": ["living room", "office room", "bedroom"],
    "computer": ["living room", "office room", "bedroom"],
    "printer": ["living room", "office room"],
    "plant": ["living room", "bedroom", "bathroom"],
    "picture": ["living room", "bedroom"],
    "painting": ["living room", "bedroom"],
    "clock": ["living room", "bedroom", "kitchen"],
    "newspaper": ["living room"],
    "magazine": ["living room"],
    "telephone": ["living room", "bedroom", "kitchen"],
    "bicycle": ["living room", "hallway", "garage"],
    
    # Dining room objects (if exists)
    "dining table": ["dining room", "living room", "kitchen"],
    "dining chair": ["dining room"],
    
    # Office/Study objects
    "desk": ["office room", "bedroom", "living room"],
    "office chair": ["office room"],
    "bookshelf": ["office room", "living room"],
    "computer": ["office room", "living room"],
    "printer": ["office room", "living room"],
    
    # Laundry objects
    "laundry machine": ["laundry room", "bathroom", "kitchen"],
    "washer": ["laundry room", "bathroom", "kitchen"],
    "dryer": ["laundry room", "bathroom", "kitchen"],
    "washing machine": ["laundry room", "bathroom", "kitchen"],
    
    # Other common objects
    "bench": ["living room", "bedroom", "hallway"],
    "stool": ["kitchen", "bedroom", "living room"],
    "cabinet": ["kitchen", "bathroom", "living room", "bedroom"],
    "display cabinet": ["living room"],
}

# Names, companies, years, materials are used to generate the user's persona
NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Fred",
    "George",
    "Helen",
    "John",
    "Kevin",
    "Lucy",
    "Peter",
    "Susan",
    "Tom",
    "Wendy",
    "Zoe",
    "Rachel",
    "Eric",
    "Olivia",
    "Dale",
    "Frank",
    "Sax Junior",
    "Sky C.",
    "Joyce",
    "Linda",
    "Rita",
]
COMPANIES = [
    "IKEA",
    "Walmart",
    "Amazon",
    "West Elm",
    "Maiden Home",
    "Pottery Barn",
    "Wayfair",
    "Castlery",
    "MoMA Design Store",
    "Pottery Barn",
    "CB2",
    "Goodee",
]
YEARS = [
    "at 2019",
    "at 2020",
    "at 2021",
    "last year",
    "this year",
    "5 uears ago",
    "10 years ago",
    "20 years ago",
    "at 2000",
    "at 1990",
    "at 1980",
    "at 2010",
]
MATERIALS = [
    "mahaogany",
    "soft steel",
    "plastic",
    "glass",
    "ceramic",
    "leather covered",
    "wool skin",
    "wood inside",
    "metal",
    "polypropylene",
    "bamboo",
    "rubber",
    "faux suede",
]
