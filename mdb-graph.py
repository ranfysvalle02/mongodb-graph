from enum import Enum
from typing import List
import json
from pymongo import MongoClient
import spacy
from openai import AzureOpenAI

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Replace with your actual values
MDB_URI = ""
MDB_DATABASE = ""
MDB_COLL = ""
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = ""
deployment_name = "gpt-4o-mini"  # The name of your model deployment

# Initialize Azure OpenAI client
az_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,api_version="2023-07-01-preview",api_key=AZURE_OPENAI_API_KEY)

class Relationship(Enum):
	WORKED_AT = "worked at"
	FOUNDED = "founded"

# List of documents to create the knowledge graph
documents = [
	"Steve Jobs founded Apple.",
	"Before Apple, Steve Jobs worked at Atari.",
	"Steve Wozniak and Steve Jobs founded Apple together.",
	"After leaving Apple, Steve Jobs founded NeXT.",
	"Steve Wozniak and Steve Jobs worked together at Apple.",
	"Bill Gates founded Microsoft.",
	"Microsoft and Apple were rivals in the early days of the personal computer market.",
	"Bill Gates worked at Microsoft for many years before stepping down as CEO.",
	"Elon Musk founded SpaceX.",
	"Before SpaceX, Elon Musk founded PayPal.",
	"Elon Musk also founded Tesla, a company that produces electric vehicles.",
	"Jeff Bezos founded Amazon.",
	"Amazon started as an online bookstore before expanding into other markets.",
	"Jeff Bezos also founded Blue Origin, a space exploration company.",
	"Blue Origin and SpaceX are competitors in the private space industry."
]

class Node:
	"""Represents a node in the knowledge graph."""
	def __init__(self, name: str, type: str):
    	self.name = name
    	self.type = type

class Edge:
	"""Represents an edge in the knowledge graph."""
	def __init__(self, source_node: Node, target_node: Node, relation: str):
    	self.source_node = source_node
    	self.target_node = target_node
    	self.relation = relation

	def __eq__(self, other):
    	if isinstance(other, Edge):
        	return self.source_node.name == other.source_node.name and self.target_node.name == other.target_node.name and self.relation == other.relation
    	return False

	def __hash__(self):
    	return hash((self.source_node.name, self.target_node.name, self.relation))

class KnowledgeGraph:
	"""Creates a knowledge graph from a list of documents."""
	def __init__(self, documents: List[str]):
    	self.documents = documents
    	self.nodes = {}
    	self.edges = []
    	# Connect to MongoDB
    	self.client = MongoClient(MDB_URI)

	def store_in_mongodb(self, db_name: str, collection_name: str):
    	"""Stores the knowledge graph in MongoDB."""
    	db = self.client[db_name]
    	collection = db[collection_name]
    	collection.delete_many({})
    	# Convert nodes and edges to a format suitable for MongoDB
    	for name, node in self.nodes.items():
        	node_data = {'_id': name, 'type': node.type, 'edges': []}
        	for edge in self.edges:
            	if edge.source_node.name == name:
                	node_data['edges'].append({'relation': edge.relation, 'target': edge.target_node.name})
        	collection.insert_one(node_data)
    
	def create_knowledge_graph(self):
    	"""Creates a knowledge graph from the list of documents."""
    	for document in self.documents:
        	relationships = []
        	prompt = f"Identify relationships in the text: ```{str(document)}```\n"
        	prompt += "Following relationships are possible: ```"
        	prompt += ", ".join([rel.value for rel in Relationship])
        	prompt += """```
	Format concise response as a JSON object with only two keys called "relationships", and "nodes".
	The value of the "relationships" key should be a list of objects each with these fields (source, source_type, relation, target, target_type).
	IF NO RELATIONSHIP IS FOUND, RETURN EMPTY LIST.
	IF NO NODES ARE FOUND, RETURN EMPTY LIST.

	[response criteria]
	- JSON object: { "relationships": [], "nodes": [] }
	- each relationship should be of the format: { "source": "Alice", "source_type": "person", "target": "MongoDB", "relation": "worked at", "target_type": "company" }
	- each node should be of the format: { "name": "MongoDB", "type": "company" }
	[end response criteria]
	"""
        	try:
            	response = az_client.chat.completions.create(
                        	model=deployment_name,
                        	messages=[
                            	{"role": "system", "content": "You are a helpful assistant that extracts the name of the person being asked about."},
                            	{"role": "system", "content": "You specialize in identifying these relationships: " + ", ".join([rel.value for rel in Relationship])},
                            	{"role": "user", "content": prompt},
                        	],
                        	response_format={ "type": "json_object" }
            	)
            	completion = json.loads(response.choices[0].message.content.strip())
           	 
            	# Parse the OpenAI response
            	for r in completion["relationships"]:
                	relationships.append((r["source"],r["source_type"],r["target"], r["relation"],r["target_type"]))
            	for n in completion["nodes"]:
                	self.nodes[n["name"]] = Node(n["name"],n["type"])
           	 
        	except Exception as e:
            	print(f"Error extracting relationships: {e}")

        	for source, source_type, target, relation, target_type in relationships:
            	if source in self.nodes and target in self.nodes:
                	edge = Edge(self.nodes[source], self.nodes[target], relation)
                	if edge not in self.edges:  # Check for duplicate edges
                    	self.edges.append(edge)

	def print_knowledge_graph(self):
    	"""Prints the nodes and edges of the knowledge graph."""
    	print("\nNodes:")
    	for node in self.nodes.values():
        	print(node.name)

    	print("\nEdges:")
    	for edge in self.edges:
        	print(f"{edge.source_node.name} {edge.relation} {edge.target_node.name}")

	def find_related_companies(self, person_name: str):
    	"""Finds companies related to a person using the knowledge graph stored in MongoDB."""
    	db = self.client["apollo-salesops"]
    	collection = db["__kg"]

    	pipeline = [
        	{
            	"$match": {
                	"_id": person_name
            	}
        	},
        	{
            	"$graphLookup": {
                	"from": "__kg",
                	"startWith": "$edges.target",
                	"connectFromField": "edges.target",
                	"connectToField": "_id",
                	"as": "related_companies",
                	"depthField": "depth"
            	}
        	},
        	{
            	"$project": {
                	"_id": 0,
                	"related_companies._id": 1,
                	"related_companies.type": 1,
                	"related_companies.depth": 1
            	}
        	}
    	]

    	result = collection.aggregate(pipeline)
    	return list(result)


# Create the knowledge graph
knowledge_graph = KnowledgeGraph(documents)
knowledge_graph.create_knowledge_graph()
knowledge_graph.print_knowledge_graph()
print("Knowledge graph created and printed.")
print("Storing knowledge graph in MongoDB.")
knowledge_graph.store_in_mongodb(MDB_DATABASE, MDB_COLL)
print("Knowledge graph stored in MongoDB.")
print("Lets begin.")
Q = "Write a rap about Elon Musk"
print("User Prompt: " + Q)
print("QUERY UNDERSTANDING: Extract the name of the person in the prompt.")
text = nlp(Q)
person = ""
for entity in text.ents:
	if entity.label_ == "PERSON":
    	print("Person: ")
    	print(entity.text.strip(',.'))
    	person = entity.text.strip(',')
    	break
print("GRAPH TRAVERSAL: Find related companies to the person.")
context_fusion = knowledge_graph.find_related_companies(person)
print("RELATED COMPANIES:")
print(context_fusion)
print("Contextual Fusion: Combines graph information with textual context.")
msgs = [
	{"role": "system", "content": "You are a helpful assistant that uses the provided additional context to generate more relevant responses."},
	{"role": "user", "content": "Given this user prompt: " + Q},
	{"role": "user", "content": "Given this additional context: ```\n" + str(context_fusion)+"\n```"},
	{"role": "user", "content": """
 	Respond to the user prompt in JSON format.
[response format]
 	- JSON object: { "response": "answer goes here" }
"""
	},
]
print(
	json.dumps(msgs, indent=2)
)
print("Language Model: Generates human-like text based on provided information.")
ai_response = az_client.chat.completions.create(model=deployment_name,
	messages=msgs,
	response_format={ "type": "json_object" }
)
ai_response = json.loads(ai_response.choices[0].message.content.strip())
print(
	ai_response.get("response")
)
