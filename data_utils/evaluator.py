class Evaluator(object):
    @staticmethod
    def check_match(ent_a, ent_b):
        return (ent_a.category == ent_b.category and
                max(ent_a.start_pos, ent_b.start_pos) < min(ent_a.end_pos, ent_b.end_pos))

    @staticmethod
    def count_intersects(ent_list_a, ent_list_b):
        num_hits = 0
        ent_list_b = ent_list_b.copy()
        for ent_a in ent_list_a:
            hit_ent = None
            for ent_b in ent_list_b:
                if Evaluator.check_match(ent_a, ent_b):
                    hit_ent = ent_b
                    break
            if hit_ent is not None:
                num_hits += 1
                ent_list_b.remove(hit_ent)
        return num_hits

    @staticmethod
    def f1_score(gt_docs, pred_docs):
        num_hits = 0
        num_preds = 0
        num_gts = 0
        for doc_id in gt_docs.doc_ids:
            gt_ents = gt_docs[doc_id].ents.ents
            pred_ents = pred_docs[doc_id].ents.ents
            num_gts += len(gt_ents)
            num_preds += len(pred_ents)
            num_hits += Evaluator.count_intersects(pred_ents, gt_ents)
        p = num_hits / num_preds
        r = num_hits / num_gts
        f = 2 * p * r / (p + r)
        return f, p, r

