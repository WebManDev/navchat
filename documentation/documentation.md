Build_graph function information: 
So, the build_graph function is called in the orion/user_simulator/topograph.py file. And in the user_simulator/base.py file, the topological graph initialization gets called. Purpose of user simulator files is to allow simulated interactions.


Where’d the paper get the LLM from? (ORION..) Is it getting something from the database?

Not from database, but it’s in the JSON files based on the scene. It's from orion/user_simulator/goals/<scene_id>/final.json


VLMap information: 

It's made in the scripts/build_vlmap.py file, which makes it per scene. It's also utilized in orion/agent_env/hybrid_search.py, so basically _prepare_vlmap() function essentially creates VLMapSearch from vlmap_path. VLMap is a pre-built, text-queryable voxel map per scene: built from recordings with LSeg or ConceptFusion, saved as sparse_vxl_map.npz, and at runtime queried by VLMapSearch to get ObjectWayPoints for navigation.

**Location:** Built by `scripts/build_vlmap.py`. Map loaded in `orion/map/map_search/search_base.py` (MapSearch._load_sparse_map); queried in `orion/map/map_search/search_voxel.py` (VLMapSearch). Used at runtime in `orion/agent_env/hybrid_search.py` (_prepare_vlmap, _vlmap_retrieve) and in `orion/agent_env/chatgpt_control_orion.py` / `orion/agent_env/chatgpt_control_vlmap.py`. User simulator loads the same map from `data/experiments/predict_<scene>_<floor>/lseg_vlmap/sparse_vxl_map.npz` in `orion/user_simulator/base.py`.



More documentation about custom functions (NOT WORKING YET... Need to tailor to current dataset and need to be tested heavily.)
orion/agent_env/chatgpt_control_orion.py is the location. 

For calculateUsingCLIP, it's basically picking the room based on using CLIP embeddings. For goToRoom, it uses a dictionary to match the desired object to the room to see which room the robot should go to. Lastly, for call LLM, an api call to ChatGPT occurs and the LLM decides which room to go to. 

I think now that I've figured out where the existing data is, I could potentially make these "custom functions" work. But it'll definitely take some time to test and re-implement. 