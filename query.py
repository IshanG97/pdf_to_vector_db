# Your query
queries = ["What does the chart on page 2 indicate?"]

# Process the query
batch_queries = processor.process_queries(queries).to(model.device)

# Compute similarity scores
with torch.no_grad():
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)

# Print results
print("Scores:", scores)
