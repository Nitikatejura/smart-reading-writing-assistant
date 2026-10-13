"""
Stage 2: Dependency Parsing and Part-of-Speech (POS) Tagging.
Produces dependency parse trees, POS sequences, and visual ASCII tree representations.
"""
import re
from typing import List, Dict, Any, Optional, Tuple


class DependencyNode:
    """Represents a token node in a dependency tree."""

    def __init__(self, token_id: int, text: str, pos: str, dep: str = "dep", head_id: int = 0):
        self.id = token_id
        self.text = text
        self.pos = pos
        self.dep = dep
        self.head_id = head_id
        self.children: List['DependencyNode'] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "pos": self.pos,
            "dep": self.dep,
            "head": self.head_id
        }


class DependencyParser:
    """
    Produces sentence segmentation, POS tagging, and typed dependency parse trees.
    Combines NLTK POS tagging with a deterministic rule-based dependency arc generator,
    with an automatic spaCy backend if available in the environment.
    """

    def __init__(self, use_spacy: bool = False):
        self.spacy_nlp = None
        if use_spacy:
            try:
                import spacy
                self.spacy_nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.spacy_nlp = None

    def segment_sentences(self, text: str) -> List[str]:
        """Splits paragraph into individual sentences cleanly."""
        text = text.strip()
        if not text:
            return []
        # Protect abbreviations like Dr., Prof.
        protected = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|vs|e\.g|i\.e)\.', r'\1__DOT__', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', protected) if s.strip()]
        return [s.replace('__DOT__', '.') for s in sentences]

    def pos_tag_sentence(self, sentence: str) -> List[Tuple[str, str]]:
        """Tags sentence tokens with Penn Treebank POS tags using NLTK."""
        tokens = re.findall(r"[A-Za-z0-9']+|[.,!?;:]", sentence)
        try:
            from nltk.tag import pos_tag
            tagged = pos_tag(tokens)
        except Exception:
            # Fallback simple POS heuristic
            tagged = [(t, "NN" if t.isalnum() else ".") for t in tokens]
        return tagged

    def parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Parses a single sentence into tokens, POS tags, and dependency tree arcs.
        """
        if self.spacy_nlp is not None:
            try:
                doc = self.spacy_nlp(sentence)
                nodes = []
                for tok in doc:
                    nodes.append({
                        "id": tok.i + 1,
                        "text": tok.text,
                        "pos": tok.tag_,
                        "dep": tok.dep_,
                        "head": tok.head.i + 1 if tok.head != tok else 0
                    })
                return {
                    "sentence": sentence,
                    "tokens": [t["text"] for t in nodes],
                    "pos_tags": [(t["text"], t["pos"]) for t in nodes],
                    "dependency_arcs": nodes,
                    "ascii_tree": self._format_spacy_tree(nodes)
                }
            except Exception:
                pass

        # Robust built-in dependency arc generator
        tagged = self.pos_tag_sentence(sentence)
        nodes: List[DependencyNode] = []
        for i, (tok, tag) in enumerate(tagged, 1):
            nodes.append(DependencyNode(token_id=i, text=tok, pos=tag))

        # Find ROOT predicate: the main finite verb (VBD, VBP, VBZ, VB)
        verb_indices = [i for i, n in enumerate(nodes) if n.pos.startswith("VB") and n.text.lower() not in {"is", "am", "are", "was", "were", "have", "had", "has", "do", "did", "can", "could", "would", "should", "will"}]
        if not verb_indices:
            verb_indices = [i for i, n in enumerate(nodes) if n.pos.startswith("VB")]
        
        root_idx = verb_indices[0] if verb_indices else (len(nodes) // 2)
        nodes[root_idx].dep = "ROOT"
        nodes[root_idx].head_id = 0

        # Assign dependencies relative to ROOT
        for i, node in enumerate(nodes):
            if i == root_idx:
                continue
            
            # Punctuation
            if node.text in {".", ",", "?", "!", ";", ":"}:
                node.dep = "punct"
                node.head_id = root_idx + 1
                continue

            # Auxiliary verbs / Modals
            if node.pos in {"MD"} or node.text.lower() in {"could", "would", "can", "should", "will", "may", "might", "must", "do", "did", "does"}:
                node.dep = "aux"
                node.head_id = root_idx + 1
                continue

            # Determiners
            if node.pos in {"DT", "PRP$"}:
                node.dep = "det"
                # Attach to following noun
                attached = False
                for j in range(i + 1, len(nodes)):
                    if nodes[j].pos.startswith("NN") or nodes[j].pos == "PRP":
                        node.head_id = j + 1
                        attached = True
                        break
                if not attached:
                    node.head_id = root_idx + 1
                continue

            # Adjectival modifier
            if node.pos.startswith("JJ"):
                node.dep = "amod"
                attached = False
                for j in range(i + 1, len(nodes)):
                    if nodes[j].pos.startswith("NN"):
                        node.head_id = j + 1
                        attached = True
                        break
                if not attached:
                    node.head_id = root_idx + 1
                continue

            # Adverbial modifier
            if node.pos.startswith("RB"):
                node.dep = "advmod"
                node.head_id = root_idx + 1
                continue

            # Prepositions
            if node.pos == "IN":
                node.dep = "prep"
                node.head_id = root_idx + 1
                # Find following noun phrase object (pobj)
                for j in range(i + 1, len(nodes)):
                    if nodes[j].pos.startswith("NN") or nodes[j].pos == "PRP":
                        nodes[j].dep = "pobj"
                        nodes[j].head_id = i + 1
                        break
                continue

            # Nouns / Pronouns: subject vs object
            if node.pos.startswith("NN") or node.pos == "PRP":
                if node.dep == "dep":  # Not yet assigned as pobj
                    if i < root_idx:
                        node.dep = "nsubj"
                        node.head_id = root_idx + 1
                    else:
                        node.dep = "dobj"
                        node.head_id = root_idx + 1

        # Format visual tree
        node_dicts = [n.to_dict() for n in nodes]
        ascii_tree = self._format_ascii_tree(node_dicts, root_idx + 1)
        nltk_tree_str = self._format_nltk_tree(node_dicts, root_idx + 1)

        return {
            "sentence": sentence,
            "tokens": [t[0] for t in tagged],
            "pos_tags": tagged,
            "dependency_arcs": node_dicts,
            "ascii_tree": ascii_tree,
            "nltk_tree": nltk_tree_str
        }

    def _format_nltk_tree(self, nodes: List[Dict[str, Any]], root_id: int) -> str:
        """Constructs an NLTK Tree formatted string and renders it using nltk.tree.Tree."""
        children_map: Dict[int, List[Dict[str, Any]]] = {}
        for n in nodes:
            children_map.setdefault(n["head"], []).append(n)

        root_node = next((n for n in nodes if n["id"] == root_id), None)
        if not root_node and nodes:
            root_node = nodes[0]

        def to_bracket(node: Dict[str, Any]) -> str:
            label = f"{node['dep'].upper()}_{node['text']}"
            my_children = children_map.get(node["id"], [])
            if not my_children:
                return f"({label} {node['pos']})"
            children_str = " ".join([to_bracket(ch) for ch in my_children])
            return f"({label} {children_str})"

        if not root_node:
            return ""

        bracket_str = to_bracket(root_node)
        try:
            from nltk.tree import Tree
            tree = Tree.fromstring(bracket_str)
            return tree.pformat()
        except Exception:
            return bracket_str

    def _format_ascii_tree(self, nodes: List[Dict[str, Any]], root_id: int) -> str:
        """Constructs an indented hierarchy tree showing heads, dependents, and relations."""
        # Map head_id -> children
        children_map: Dict[int, List[Dict[str, Any]]] = {}
        for n in nodes:
            children_map.setdefault(n["head"], []).append(n)

        root_node = next((n for n in nodes if n["id"] == root_id), None)
        if not root_node and nodes:
            root_node = nodes[0]

        lines = []
        def render(node: Dict[str, Any], depth: int = 0, is_last: bool = True):
            indent = "  " * depth
            branch = "\\-- " if is_last else "+-- "
            lines.append(f"{indent}{branch}[{node['dep'].upper()}] {node['text']} ({node['pos']})")
            my_children = children_map.get(node["id"], [])
            for idx, ch in enumerate(my_children):
                render(ch, depth + 1, idx == len(my_children) - 1)

        if root_node:
            lines.append(f"ROOT: {root_node['text']} ({root_node['pos']})")
            my_children = children_map.get(root_node["id"], [])
            for idx, ch in enumerate(my_children):
                render(ch, 1, idx == len(my_children) - 1)

        return "\n".join(lines)

    def _format_spacy_tree(self, nodes: List[Dict[str, Any]]) -> str:
        root = next((n for n in nodes if n["dep"] == "ROOT"), nodes[0] if nodes else None)
        root_id = root["id"] if root else 0
        return self._format_ascii_tree(nodes, root_id)

    def parse_paragraph(self, text: str) -> List[Dict[str, Any]]:
        """Parses all sentences in a paragraph."""
        sentences = self.segment_sentences(text)
        return [self.parse_sentence(s) for s in sentences]
