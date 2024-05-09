docker run -d --name inference-container --mount type=bind,source=F:/grammar_ru/data-cache/downloads/models,target=/models -p 80:80 inference-image 
