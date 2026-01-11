#!/usr/bin/env python3
"""
Analyze relationship density in HotpotQA documents

Manually count explicit relationships in sample documents
to see if the dataset actually contains enough relationships.
"""

import json

# Load first 3 documents
with open('data/hotpotqa/test_dataset/test_corpus.json', 'r') as f:
    corpus = json.load(f)

print("\n" + "="*70)
print("MANUAL RELATIONSHIP ANALYSIS - HotpotQA Documents")
print("="*70)

# Document 1: Ed Wood (film)
print("\n" + "-"*70)
print("DOCUMENT 1: Ed Wood (1994 film)")
print("-"*70)
doc1 = corpus[0]['text']
print(doc1[:600])

print("\n\nMANUAL RELATIONSHIP COUNT:")
relationships_doc1 = [
    ("Ed Wood (film)", "DIRECTED_BY", "Tim Burton"),
    ("Ed Wood (film)", "PRODUCED_BY", "Tim Burton"),
    ("Ed Wood (film)", "STARS", "Johnny Depp"),
    ("Johnny Depp", "PLAYS", "Ed Wood"),
    ("Ed Wood (film)", "ABOUT", "Ed Wood"),
    ("Ed Wood (film)", "FEATURES", "Bela Lugosi"),
    ("Bela Lugosi", "PLAYED_BY", "Martin Landau"),
    ("Ed Wood (film)", "STARS", "Sarah Jessica Parker"),
    ("Ed Wood (film)", "STARS", "Patricia Arquette"),
    ("Ed Wood (film)", "STARS", "Jeffrey Jones"),
    ("Ed Wood (film)", "STARS", "Lisa Marie"),
    ("Ed Wood (film)", "STARS", "Bill Murray"),
    ("Ed Wood (film)", "RELEASED_IN", "1994"),
]

for i, (src, rel, tgt) in enumerate(relationships_doc1, 1):
    print(f"{i:2d}. {src} --[{rel}]--> {tgt}")

print(f"\nTotal explicit relationships in Doc 1: {len(relationships_doc1)}")

# Document 2: Scott Derrickson
print("\n" + "-"*70)
print("DOCUMENT 2: Scott Derrickson")
print("-"*70)
doc2 = corpus[1]['text']
print(doc2[:500])

print("\n\nMANUAL RELATIONSHIP COUNT:")
relationships_doc2 = [
    ("Scott Derrickson", "BORN_ON", "July 16, 1966"),
    ("Scott Derrickson", "NATIONALITY", "American"),
    ("Scott Derrickson", "OCCUPATION", "director"),
    ("Scott Derrickson", "OCCUPATION", "screenwriter"),
    ("Scott Derrickson", "OCCUPATION", "producer"),
    ("Scott Derrickson", "LIVES_IN", "Los Angeles"),
    ("Los Angeles", "LOCATED_IN", "California"),
    ("Scott Derrickson", "DIRECTED", "Sinister"),
    ("Sinister", "GENRE", "horror"),
    ("Scott Derrickson", "DIRECTED", "The Exorcism of Emily Rose"),
    ("The Exorcism of Emily Rose", "GENRE", "horror"),
    ("Scott Derrickson", "DIRECTED", "Deliver Us From Evil"),
    ("Deliver Us From Evil", "GENRE", "horror"),
    ("Scott Derrickson", "DIRECTED", "Doctor Strange"),
    ("Doctor Strange", "PART_OF", "Marvel Cinematic Universe"),
    ("Doctor Strange", "RELEASED_IN", "2016"),
]

for i, (src, rel, tgt) in enumerate(relationships_doc2, 1):
    print(f"{i:2d}. {src} --[{rel}]--> {tgt}")

print(f"\nTotal explicit relationships in Doc 2: {len(relationships_doc2)}")

# Document 3: Woodson
print("\n" + "-"*70)
print("DOCUMENT 3: Woodson (location)")
print("-"*70)
doc3 = corpus[2]['text']
print(doc3[:500])

print("\n\nMANUAL RELATIONSHIP COUNT:")
relationships_doc3 = [
    ("Woodson", "TYPE", "census-designated place"),
    ("Woodson", "LOCATED_IN", "Pulaski County"),
    ("Pulaski County", "LOCATED_IN", "Arkansas"),
    ("Arkansas", "LOCATED_IN", "United States"),
    ("Woodson", "POPULATION", "403"),
    ("Woodson", "PART_OF", "Little Rock–North Little Rock–Conway Metropolitan Statistical Area"),
    ("Woodson", "NAMED_AFTER", "Ed Wood Sr."),
    ("Woodson Lake", "NAMED_AFTER", "Ed Wood Sr."),
    ("Wood Hollow", "NAMED_AFTER", "Ed Wood Sr."),
    ("Ed Wood Sr.", "OCCUPATION", "plantation owner"),
    ("Ed Wood Sr.", "OCCUPATION", "trader"),
    ("Ed Wood Sr.", "OCCUPATION", "businessman"),
    ("Woodson", "ADJACENT_TO", "Wood Plantation"),
    ("Wood Plantation", "OWNED_BY", "Ed Wood Sr."),
]

for i, (src, rel, tgt) in enumerate(relationships_doc3, 1):
    print(f"{i:2d}. {src} --[{rel}]--> {tgt}")

print(f"\nTotal explicit relationships in Doc 3: {len(relationships_doc3)}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

total_relationships = len(relationships_doc1) + len(relationships_doc2) + len(relationships_doc3)
print(f"\nTotal relationships across 3 documents: {total_relationships}")
print(f"Average relationships per document: {total_relationships/3:.1f}")

print("\n" + "="*70)
print("BASELINE VS POTENTIAL")
print("="*70)

print(f"\nIf baseline extracted: 1.17 relationships per node")
print(f"Manual count shows: ~{total_relationships/3:.1f} relationships per document")
print("\n⚠️  BASELINE IS MISSING MOST RELATIONSHIPS!")

print("\nConclusion:")
print("✅ HotpotQA documents ARE rich in explicit relationships")
print("✅ The problem is extraction quality, NOT dataset quality")
print("✅ KARMA's multi-label approach should find these relationships")

print("\n" + "="*70)
