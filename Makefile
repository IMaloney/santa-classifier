
# RUN = 

# data_repl:
# 	conda activate ml
# 	python3 code/data_collection/data_collection_repl.py
# 	conda deactivate ml

# convert_model:
# 	conda activate ml
# 	python3 scripts/convert_model.py
# 	conda deactivate ml

# cleanup_log_folders:
# 	conda activate ml
# 	python3 scripts/cleanup.py -a
# 	conda deactivate ml

# delete_run:
# 	conda activate ml 
# 	python3 scripts/cleanup.py -r $(RUN)
# 	conda deactivate ml

# run_checkpoint:
# 	conda activate ml
# 	python3 code/model/main.py 
# 	conda deactivate ml

# make_graph:
# 	conda activate ml 
# 	python3 scripts/create_graphs.py -r $(RUN)
# 	conda deactivate ml 