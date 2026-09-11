import json

def main():
    with open('data/evaluation/results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=========================================================================================")
    print("                       SUTRA MULTIMODAL RAG - RETRIEVAL EVALUATION RESULTS               ")
    print("=========================================================================================\n")

    summary = data.get('summary_metrics', {})
    for mode_key, metrics in summary.items():
        mode_label = mode_key.upper().replace('_', ' ')
        print(f"Mode: {mode_label}")
        print(f"  • Doc Top-1 Accuracy : {metrics['doc_top1_accuracy']*100:.2f}%")
        print(f"  • Page Hit@1         : {metrics['page_hit_at_1']*100:.2f}%")
        print(f"  • Page Hit@3         : {metrics['page_hit_at_3']*100:.2f}%")
        print(f"  • Page Hit@5         : {metrics['page_hit_at_5']*100:.2f}%")
        print(f"  • Page MRR           : {metrics['page_mrr']:.4f}")
        print(f"  • Avg Latency        : {metrics['avg_retrieval_latency_ms']:.2f} ms")
        print("-" * 75)

    print("\n=========================================================================================")
    print("                              QUERY-BY-QUERY DETAILED RESULTS                             ")
    print("=========================================================================================\n")

    results = data.get('results', [])
    for q in results:
        qid = q['query_id']
        question = q['question']
        expected_doc = q['expected_document']
        expected_page = q['expected_page']
        is_neg = q['is_negative']

        exp_str = "None (Unanswerable)" if is_neg else f"{expected_doc} (Page {expected_page})"
        print(f"[{qid}] Query: \"{question}\"")
        print(f"     Expected Target: {exp_str}")

        for mode_name, res in q['modes'].items():
            top = res.get('top_result', {})
            top_doc = top.get('notice_id', 'N/A')
            top_page = top.get('page_number', 'N/A')
            d_ok = "PASS" if res.get('correct_document_retrieved') else "FAIL"
            p_ok = "PASS" if res.get('hit1_page') else "FAIL"
            p_rank = res.get('page_rank', 'N/A')
            time_ms = res.get('retrieval_time_ms', 0)

            print(f"     -> {mode_name:<20} | Doc Match: {d_ok:<4} | Page Hit@1: {p_ok:<4} (Rank {p_rank}) | Top Output: {top_doc} p.{top_page} | Latency: {time_ms:.1f}ms")
        print("-" * 90)

if __name__ == '__main__':
    main()
