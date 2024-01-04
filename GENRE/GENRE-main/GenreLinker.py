import pickle

#from genre.fairseq_model import GENRE
from genre.hf_model import GENRE as genre_hf
from genre.trie import Trie
class GenreLinker:
    def __init__(self):
        with open("text_to_wikidata_id", "rb") as f:
            self.title_to_wikidata_id = pickle.load(f)
        # load the prefix tree (trie)
        with open("kilt_titles_trie_dict.pkl", "rb") as f:
            self.trie = Trie.load_from_dict(pickle.load(f))

        # load the model
        self.model = genre_hf.from_pretrained("hf_entity_disambiguation_aidayago").eval()

    def link_entities(self,texts_with_marked_entities:str):
        #entities should be marked like [START_ENT]Einstein[END_ENT] was a German physicist.
        links=self.model.sample(
            sentences=["[START_ENT]Einstein[END_ENT] was a German physicist."],
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
        )

        for el in links[0]:
            el["wikidata_id"]=self.title_to_wikidata_id[el["text"]]
        return links