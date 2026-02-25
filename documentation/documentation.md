Build_graph function information: 
So, the build_graph function is called in the orion/user_simulator/topograph.py file. And in the user_simulator/base.py file, the topological graph initialization gets called. Purpose of user simulator files is to allow simulated interactions.


Where’d the paper get the LLM from? (ORION..) Is it getting something from the database?

Not from database, but it’s in the JSON files based on the scene. It's from orion/user_simulator/goals/<scene_id>/final.json


VLMap information: 

It's made in the scripts/build_vlmap.py file, which makes it per scene. It's also utilized in orion/agent_env/hybrid_search.py, so basically _prepare_vlmap() function essentially creates VLMapSearch from vlmap_path. VLMap is a pre-built, text-queryable voxel map per scene: built from recordings with LSeg or ConceptFusion, saved as sparse_vxl_map.npz, and at runtime queried by VLMapSearch to get ObjectWayPoints for navigation.

**Location:** Built by `scripts/build_vlmap.py`. Map loaded in `orion/map/map_search/search_base.py` (MapSearch._load_sparse_map); queried in `orion/map/map_search/search_voxel.py` (VLMapSearch). Used at runtime in `orion/agent_env/hybrid_search.py` (_prepare_vlmap, _vlmap_retrieve) and in `orion/agent_env/chatgpt_control_orion.py` / `orion/agent_env/chatgpt_control_vlmap.py`. User simulator loads the same map from `data/experiments/predict_<scene>_<floor>/lseg_vlmap/sparse_vxl_map.npz` in `orion/user_simulator/base.py`.



More documentation about custom functions (now using scene data from final.json)
orion/agent_env/chatgpt_control_orion.py is the location. 

Data source: orion/user_simulator/goals/<scene_id>/final.json provides per-scene object→room mapping. Falls back to OBJECT_TO_ROOMS in my_config.py when final.json is unavailable.

For calculateUsingCLIP, it picks the room using CLIP embeddings, with a boost for rooms that contain the target according to final.json. For goToRoom, it uses the object→room mapping from final.json (or OBJECT_TO_ROOMS fallback) to determine which room to navigate to. For callLLM, an API call to ChatGPT occurs with scene-based room suggestions, and the LLM decides which room to go to. 

Going to ORION/config/myconfig.py, the number grid is 600, whilst the height is 40 voxels. and the cell size is .05 meters. So for the 2D occupancy map, it's 30x30m and for 3D VLMap voxel grid, it's 30x30x2