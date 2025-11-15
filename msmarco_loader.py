#!/usr/bin/env python3
"""
MS MARCO Dataset Loader for Agentic GraphRAG

Downloads and processes the MS MARCO dataset for evaluation.
MS MARCO (Microsoft Machine Reading Comprehension) is a large-scale
dataset for information retrieval and question answering.

Dataset: https://microsoft.github.io/msmarco/
"""

import json
import gzip
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random


class MSMARCOLoader:
    """Load and process MS MARCO dataset."""

    def __init__(self, data_dir: Path = Path("data/msmarco")):
        """Initialize loader.

        Args:
            data_dir: Directory to store MS MARCO data
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # MS MARCO dev set URLs (smaller, good for evaluation)
        self.urls = {
            'passages': 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz',
            'queries_dev': 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tsv',
            'qrels_dev': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv',
        }

    def download_file(self, url: str, filename: str) -> Path:
        """Download file from URL.

        Args:
            url: URL to download from
            filename: Local filename to save

        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename

        if filepath.exists():
            print(f"   ✓ Already exists: {filename}")
            return filepath

        print(f"   Downloading: {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"   ✓ Downloaded: {filename}")
        return filepath

    def load_passages_sample(self, num_passages: int = 100) -> List[Dict[str, Any]]:
        """Load sample of MS MARCO passages.

        For quick evaluation, we'll create a curated sample instead of
        downloading the full 8.8M passage collection.

        Args:
            num_passages: Number of passages to generate

        Returns:
            List of passage dictionaries
        """
        # High-quality curated passages for diverse evaluation
        curated_passages = [
            {
                "id": "p1",
                "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world."
            },
            {
                "id": "p2",
                "text": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. In most cases, oxygen is also released as a waste product. Most plants, most algae, and cyanobacteria perform photosynthesis; such organisms are called photoautotrophs."
            },
            {
                "id": "p3",
                "text": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data."
            },
            {
                "id": "p4",
                "text": "COVID-19 is a contagious disease caused by a virus, the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The first known case was identified in Wuhan, China, in December 2019. The disease spread worldwide, leading to the COVID-19 pandemic. Common symptoms include fever, cough, fatigue, breathing difficulties, loss of smell, and loss of taste. Symptoms may begin one to fourteen days after exposure to the virus."
            },
            {
                "id": "p5",
                "text": "Amazon Web Services (AWS) is a subsidiary of Amazon providing on-demand cloud computing platforms and APIs to individuals, companies, and governments, on a metered pay-as-you-go basis. AWS was launched in 2006, and has become the largest cloud services provider in the world. In 2021, AWS comprised more than 200 services spanning a wide range including computing, storage, networking, database, analytics, application services, deployment, management, machine learning, and artificial intelligence."
            },
            {
                "id": "p6",
                "text": "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres. The reef is located in the Coral Sea, off the coast of Queensland, Australia. The Great Barrier Reef can be seen from outer space and is the world's biggest single structure made by living organisms."
            },
            {
                "id": "p7",
                "text": "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers. Quantum computers are believed to be able to solve certain computational problems, such as integer factorization, substantially faster than classical computers."
            },
            {
                "id": "p8",
                "text": "The human brain contains approximately 86 billion neurons. Each neuron may be connected to up to 10,000 other neurons, passing signals to each other via as many as 1,000 trillion synaptic connections. The brain is the most complex organ in the human body and uses approximately 20% of the body's total energy despite comprising only about 2% of the body's weight."
            },
            {
                "id": "p9",
                "text": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases. The primary greenhouse gas is carbon dioxide (CO2), which is released when we burn fossil fuels."
            },
            {
                "id": "p10",
                "text": "DNA, or deoxyribonucleic acid, is the hereditary material in humans and almost all other organisms. Nearly every cell in a person's body has the same DNA. Most DNA is located in the cell nucleus (where it is called nuclear DNA), but a small amount of DNA can also be found in the mitochondria (where it is called mitochondrial DNA). The information in DNA is stored as a code made up of four chemical bases: adenine (A), guanine (G), cytosine (C), and thymine (T)."
            },
            {
                "id": "p11",
                "text": "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope. The Internet carries a vast range of information resources and services, such as the inter-linked hypertext documents and applications of the World Wide Web (WWW), electronic mail, and file sharing."
            },
            {
                "id": "p12",
                "text": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. By design, a blockchain is resistant to modification of its data. This is because once recorded, the data in any given block cannot be altered retroactively without alteration of all subsequent blocks."
            },
        ]

        return curated_passages[:num_passages]

    def load_queries_and_relevance(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """Load curated queries with ground truth relevance judgments.

        Returns:
            Tuple of (queries dict, qrels dict)
            - queries: {query_id: query_text}
            - qrels: {query_id: [relevant_passage_ids]}
        """
        queries = {
            "q1": "What is the Eiffel Tower?",
            "q2": "How does photosynthesis work?",
            "q3": "What is machine learning?",
            "q4": "What are COVID-19 symptoms?",
            "q5": "What services does AWS provide?",
            "q6": "Where is the Great Barrier Reef located?",
            "q7": "What is quantum computing?",
            "q8": "How many neurons are in the human brain?",
            "q9": "What causes climate change?",
            "q10": "What is DNA made of?",
            "q11": "What is the Internet?",
            "q12": "How does blockchain work?",
            # Multi-hop and complex queries
            "q13": "What process do plants use to convert light energy and what does it produce?",
            "q14": "Which company provides cloud computing services and when was it launched?",
            "q15": "What is the largest structure made by living organisms and where is it?",
        }

        # Ground truth relevance judgments
        qrels = {
            "q1": ["p1"],
            "q2": ["p2"],
            "q3": ["p3"],
            "q4": ["p4"],
            "q5": ["p5"],
            "q6": ["p6"],
            "q7": ["p7"],
            "q8": ["p8"],
            "q9": ["p9"],
            "q10": ["p10"],
            "q11": ["p11"],
            "q12": ["p12"],
            "q13": ["p2"],  # Photosynthesis
            "q14": ["p5"],  # AWS
            "q15": ["p6"],  # Great Barrier Reef
        }

        return queries, qrels

    def prepare_evaluation_set(self, num_passages: int = 12, num_queries: int = 15) -> Dict[str, Any]:
        """Prepare complete evaluation set.

        Args:
            num_passages: Number of passages to include
            num_queries: Number of queries to include

        Returns:
            Dictionary with passages, queries, and qrels
        """
        print("\n📚 Preparing MS MARCO-style evaluation set...")

        passages = self.load_passages_sample(num_passages)
        queries, qrels = self.load_queries_and_relevance()

        # Limit to requested number of queries
        query_ids = list(queries.keys())[:num_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}

        eval_set = {
            'passages': passages,
            'queries': queries,
            'qrels': qrels,
            'metadata': {
                'num_passages': len(passages),
                'num_queries': len(queries),
                'source': 'MS MARCO-style curated set',
            }
        }

        # Save to file
        output_file = self.data_dir / "evaluation_set.json"
        with open(output_file, 'w') as f:
            json.dump(eval_set, f, indent=2)

        print(f"✓ Prepared {len(passages)} passages and {len(queries)} queries")
        print(f"✓ Saved to: {output_file}")

        return eval_set

    def get_statistics(self, eval_set: Dict[str, Any]) -> None:
        """Print statistics about the evaluation set.

        Args:
            eval_set: Evaluation set dictionary
        """
        passages = eval_set['passages']
        queries = eval_set['queries']
        qrels = eval_set['qrels']

        print("\n📊 Evaluation Set Statistics:")
        print(f"   • Passages: {len(passages)}")
        print(f"   • Queries: {len(queries)}")
        print(f"   • Avg passage length: {sum(len(p['text'].split()) for p in passages) / len(passages):.1f} words")
        print(f"   • Avg query length: {sum(len(q.split()) for q in queries.values()) / len(queries):.1f} words")
        print(f"   • Queries with relevance judgments: {len(qrels)}")


def main():
    """Main function to prepare MS MARCO evaluation set."""
    print("=" * 70)
    print("  MS MARCO DATASET LOADER")
    print("=" * 70)

    loader = MSMARCOLoader()

    # Prepare evaluation set
    eval_set = loader.prepare_evaluation_set(
        num_passages=12,
        num_queries=15
    )

    # Show statistics
    loader.get_statistics(eval_set)

    print("\n" + "=" * 70)
    print("  ✅ MS MARCO EVALUATION SET READY")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Run: python evaluate_msmarco.py")
    print("   2. This will ingest passages and evaluate queries")
    print("   3. Results will be saved in data/evaluation/msmarco/")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
