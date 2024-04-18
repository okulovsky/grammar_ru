from typing import List

from ..data_structures.api_data import Paragraph, ParagraphType, ResponseBody


class ResponseConstructor:
    def __init__(self) -> None:
        pass

    def construct_response(self, paragraphs: List[Paragraph], finalized_paragraph: str) -> ResponseBody:
        if finalized_paragraph == "":
            raise ValueError("Finalized paragraph is empty")
        
        if len(paragraphs) == 0:
            raise ValueError("Paragraphs list is empty")
        
        new_paragraph = Paragraph(
            type=ParagraphType.FINAL,
            is_target=False,
            content=finalized_paragraph,
            sub_paragraphs=[]
        )

        new_list = self.__reconstruct_paragraphs(paragraphs, new_paragraph)

        return ResponseBody(paragraphs=new_list)

    def __reconstruct_paragraphs(self, paragraphs: List[Paragraph], new_paragraph):
        paragraph_is_inserted = False

        def recursive_reconstruct(paragraphs: List[Paragraph], cur_paragraph: Paragraph):
            nonlocal paragraph_is_inserted
            cur_subparagraphs = []

            for idx in range(len(paragraphs)):
                if paragraphs[idx].is_target:
                    if not paragraph_is_inserted:
                        paragraph_is_inserted = True
                        cur_subparagraphs.append(new_paragraph)
                else:
                    cur_subparagraphs.append(paragraphs[idx])
                recursive_reconstruct(paragraphs[idx].sub_paragraphs, paragraphs[idx])
            
            if cur_paragraph != None:
                cur_paragraph.sub_paragraphs = cur_subparagraphs

            return cur_subparagraphs
        
        return recursive_reconstruct(paragraphs, None)