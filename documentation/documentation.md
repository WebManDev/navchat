Build_graph function information: 
So, the build_graph function is called in the orion/user_simulator/topograph.py file. And in the user_simulator/base.py file, the topological graph initialization gets called. Purpose of user simulator files is to allow simulated interactions.


Where’d the paper get the LLM from? (ORION..) Is it getting something from the database?

Not from database, but it’s in the JSON files based on the scene. It's from orion/user_simulator/goals/<scene_id>/final.json


VLMap information: 

It's made in the scripts/build_vlmap.py file, which makes it per scene. It's also utilized in orion/agent_env/hybrid_search.py, so basically _prepare_vlmap() function essentially creates VLMapSearch from vlmap_path. 