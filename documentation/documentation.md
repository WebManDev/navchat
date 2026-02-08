Build_graph function information: 
So, the build_graph function is called in the orion/user_simulator/topograph.py file. And in the user_simulator/base.py file, the topological graph initialization gets called. Purpose of user simulator files is to allow simulated interactions.


Where’d the paper get the LLM from? (ORION..) Is it getting something from the database?

Not from database, but it’s in the JSON files based on the scene. It's from orion/user_simulator/goals/<scene_id>/final.json


VLMap information: 

It's made in the scripts/build_vlmap.py file, which makes it per scene. It's also utilized in orion/agent_env/hybrid_search.py, so basically _prepare_vlmap() function essentially creates VLMapSearch from vlmap_path. VLMap is a pre-built, text-queryable voxel map per scene: built from recordings with LSeg or ConceptFusion, saved as sparse_vxl_map.npz, and at runtime queried by VLMapSearch to get ObjectWayPoints for navigation.

**Location:** Built by `scripts/build_vlmap.py`. Map loaded in `orion/map/map_search/search_base.py` (MapSearch._load_sparse_map); queried in `orion/map/map_search/search_voxel.py` (VLMapSearch). Used at runtime in `orion/agent_env/hybrid_search.py` (_prepare_vlmap, _vlmap_retrieve) and in `orion/agent_env/chatgpt_control_orion.py` / `orion/agent_env/chatgpt_control_vlmap.py`. User simulator loads the same map from `data/experiments/predict_<scene>_<floor>/lseg_vlmap/sparse_vxl_map.npz` in `orion/user_simulator/base.py`.