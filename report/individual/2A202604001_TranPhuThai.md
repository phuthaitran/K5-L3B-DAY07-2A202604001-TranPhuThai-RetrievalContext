# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk:
    - Nguồn: DSA.pdf trong repo day 6 
    - Giấy phép: CC BY-NC-SA
    - Số trang: 53 trang
    - Số chunk: 238 chunks
- Vector store, embedding model, phiên bản và số chiều:
    - **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, Sentence Transformers 6.3.0, chạy CPU; độ dài đầu vào tối đa ghi nhận là 256 token. Revision ghi là mặc định Hugging Face, không có commit hash.
    - **Vector store:** PostgreSQL 16.16 + pgvector 0.8.7; exact cosine search, không dùng approximate vector index.
- Agent model, cách chạy và phân chia dev/test:
    - **Agent model:** Google `gemini-3.8-flash` qua SDK `google-genai` 3.3.0, native function calling, `temperature=0`, thinking budget mặc định của model. Model được đọc từ biến `GEMINI_MODEL` trong `.env`.
    - **Kết nối Day 6:** `src/retrievers.py` mở kết nối tới bảng `dsa_chunks` (Strategy B) bằng `DATABASE_URL`; query được embed bằng đúng model Day 6 (kiểm tra số chiều = 384 khi khởi tạo, sai thì báo lỗi). `data/chunks.json` là bản xuất 238 chunk từ cùng bảng này, dùng cho BM25/TF-IDF và để chấm.
    - **Phân chia:** 24 câu = dev 17 (15 có đáp án + 2 thiếu evidence) và test 7 (5 có đáp án + 2 thiếu evidence). 20 câu có đáp án lấy từ benchmark Day 6, giữ nguyên split Day 6. Mọi thử nghiệm và lựa chọn chỉ dùng dev; test chạy một lần sau khi chốt cấu hình (xem `results/selection_criteria.md`).
    - **Cách chạy:** xem mục "Đường dẫn trace, nhãn, lệnh chạy" bên dưới và `src/README.md`.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm:
    - **Câu có đáp án:** mỗi câu có `expected_answer` và `gold = {source_id, page, evidence}`. `scripts/prepare_data.py` kiểm tra evidence nằm **nguyên văn** trong một chunk đúng nguồn và trang, sai thì dừng. Hai câu q02 và q11 trong Day 6 bị mất xuống dòng trong `evidence_quote` nên được thay bằng đoạn văn y hệt nhưng giữ nguyên xuống dòng của chunk (không đổi nội dung).
    - **Câu thiếu evidence:** chọn câu **gần chủ đề** (hash table, đường đi ngắn nhất, Turing machine, merge sort) nhưng hỏi chi tiết tài liệu không có: load factor/resize hash table, thuật toán thay Dijkstra khi cạnh âm, chứng minh halting problem không quyết định được, tính ổn định của merge sort. Sự vắng mặt được kiểm tra bằng tìm kiếm từ khóa không phân biệt hoa thường trên **toàn bộ 238 chunk** (không dựa vào top-k), ví dụ `load factor`, `resiz`, `rehash`, `bellman`, `halting`, `undecid`, `stability`… đều 0 kết quả. Kết quả kiểm tra lưu trong trường `absence_check` của `data/agent_cases.json`.
    - **Chấm retrieval/context:** một chunk là hit nếu đúng `source_id`, đúng `page` và chứa nguyên văn evidence (`starter/baseline.hit`). Context Evidence Rate = số câu có chunk hit trong **context cuối gửi model** / số câu có đáp án; câu thiếu evidence không vào mẫu số.
    - **Chấm agent:** theo BENCHMARK.md. Grounded answer 0/1/2 (từ chối câu có đáp án = 0); mỗi citation được đọc đối chiếu với chunk xem có hỗ trợ đúng claim gắn với nó không (ID hợp lệ chưa đủ); câu thiếu evidence đúng khi agent từ chối và không đưa khẳng định vô căn cứ. Context trên test chấm 0/1/2 theo §2.

## Retrieval và thử nghiệm

Điều kiện chung: cùng corpus 238 chunk, cùng index pgvector, cùng 15 câu dev có đáp án, query gốc (không viết lại) cho Hit@1 / Evidence Hit@3, top-3, không rerank. `filter` chỉ dùng khi câu hỏi có phạm vi section do người dùng nêu (q03, q14, q17, q18). Context budget đo bằng **ký tự** của context đã đóng gói, kể cả header `[alias | nguồn | trang | section]`. Agent: tối đa 3 lượt search, top-k 3 mỗi lượt.

Cột Context Evidence Rate: số đầu là **context cuối của agent thật** (có thể gồm nhiều lượt search); trong ngoặc là context một lượt search với query gốc ở cùng budget (`results/retrieval/summary_*.json`). Thời gian: retrieval là trung vị ms/query; agent là trung vị giây/câu và tổng thời gian cả split. Chi phí: retrieval chạy cục bộ (0 USD); agent ghi token đầu vào / token đầu ra + thinking. Chưa quy đổi USD vì chưa nhập đơn giá `gemini-3.8-flash` (có thể điền `GEMINI_PRICE_*` trong `.env` để script tự tính).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Tham chiếu TF-IDF (starter) | TF-IDF, top-3, filter section, không rerank, 2500 ký tự | Mốc lexical của starter, chỉ đo retrieval | dev | 9/15 (60,0%) | 11/15 (73,3%) | – (11/15 một lượt) | 1,7 ms; 0 USD |
| Baseline | Dense pgvector Day 6 (cosine), top-3, filter section, không rerank, 2500 ký tự, context theo thứ tự lượt search | Mốc chuẩn dùng vector store Day 6 | dev | 7/15 (46,7%) | 14/15 (93,3%) | 10/15 (66,7%) (12/15) | 18,7 ms; agent 6,65 s/câu, 162 s; 41,3k / 9,2k token |
| Phương pháp 2 | BM25 (k1=1,5, b=0,75), top-3, filter section, 2500 ký tự | Câu hỏi dùng thuật ngữ trùng sách nên lexical đưa evidence lên top-1 tốt hơn dense | dev | 12/15 (80,0%) | 12/15 (80,0%) | 12/15 (80,0%) (12/15) | 1,6 ms; agent 7,33 s/câu, 130 s; 35,9k / 7,0k token |
| Phương pháp 3 | Hybrid RRF: dense + BM25, mỗi bên 10 candidate, k=60, trọng số 0,5/0,5, top-3, 2500 ký tự | Dense mạnh ở recall@3, BM25 mạnh ở top-1; hợp nhất bằng rank nên không cần chuẩn hóa score | dev | 12/15 (80,0%) | 14/15 (93,3%) | 14/15 (93,3%) (14/15) | 18,7 ms; agent 6,96 s/câu, 151 s; 35,0k / 7,8k token |
| Cải thiện 1 | Hybrid RRF như trên, 2500 ký tự, **context xếp theo rank** (top-1 của mọi lượt search trước, rồi top-2…) | Lỗi dev 1: lượt search đầu lấp đầy budget nên chunk tốt của lượt sau không vào context; xếp theo rank sẽ cho chunk top-1 của lượt sau vào được | dev | 12/15 (80,0%) | 14/15 (93,3%) | 13/15 (86,7%) | 18,7 ms; agent 5,94 s/câu, 143 s; 35,6k / 7,0k token |
| Cải thiện 2 (đã chọn) | Hybrid RRF top-3 + **chunk lân cận ±1** (cùng nguồn, theo thứ tự ký tự trong trang), 5000 ký tự, context theo thứ tự lượt search | Lỗi dev 2: chunking Day 6 cắt đôi đoạn văn, evidence nằm ở chunk kề với chunk được tìm thấy | dev | 12/15 (80,0%) | 14/15 (93,3%) | 15/15 (100%) (15/15) | 37,9 ms*; agent 6,40 s/câu, 149 s; 49,5k / 8,2k token |
| Cấu hình đã chọn | Như Cải thiện 2 | Chạy một lần sau khi chốt | test | 4/5 (80,0%) | 5/5 (100%) | 5/5 (100%) (5/5) | 39,2 ms*; agent 8,35 s/câu, 60 s; 20,7k / 3,2k token |
| Baseline | Như Baseline dev | Chạy một lần sau khi chốt | test | 3/5 (60,0%) | 5/5 (100%) | 4/5 (80,0%) (5/5) | 17,9 ms; agent 7,89 s/câu, 60 s; 16,8k / 2,8k token |

\* Script benchmark gọi pipeline neighbor hai lần (top-10 để xếp hạng, top-3 để đóng gói context) nên thời gian retrieval gấp đôi; bản thân việc thêm chunk lân cận chỉ là tra cứu trong bộ nhớ.

Ghi chú: Cải thiện 2 vừa là một pipeline retrieval khác (thêm chunk lân cận) vừa là cấu hình cải thiện. Nó không đổi thứ hạng nên Hit@1/Hit@3 giữ nguyên như hybrid, chỉ đổi context. Để tách ảnh hưởng của budget, hybrid thường ở cùng budget 5000 vẫn chỉ đạt 14/15, còn hybrid + lân cận đạt 15/15; vậy phần tăng là do chunk lân cận. Ở budget 800 mọi phương pháp chỉ đạt 2–3/15 vì một chunk trung bình dài 722 ký tự, nên 800 ký tự chỉ chứa được khoảng một chunk.

**Hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:**

1. **Lỗi 1: context bão hòa gây từ chối nhầm** (Baseline dense: q02, q12, q13; BM25: q12). Đọc trace q12 (dense): lượt search thứ 3 (`binary search log N sequential search O(N)`) trả về chunk gold trang 21 ở **rank 1**, nhưng context 2500 ký tự đã đầy bởi các chunk trang 29, 51 và 19 từ hai lượt trước, nên chunk gold không bao giờ tới model và agent từ chối. Ở q02 và q13, chunk gold ở rank 3 của lượt đầu và bị loại vì hai chunk đứng trước đã dùng hết budget. Nguyên nhân là cách đóng gói của starter: giữ chunk của lượt trước, chunk không vừa thì bỏ. Kết quả là lượt search thêm tốn token mà không thêm được bằng chứng.
   - *Cách thử:* Cải thiện 1 (xếp context theo rank). *Kết quả:* grounded trung bình bằng hybrid (1,93 so với 1,93), thời gian giảm nhẹ (5,94 so với 6,96 s/câu), Context Evidence Rate 13/15 so với 14/15. Trace q04 cho thấy cơ chế hoạt động đúng: chunk AVL trang 6 (rank 1 ở lượt 2) vào được context. Tuy vậy **không có cải thiện đo được về chất lượng** trên dev, vì với hybrid, gold thường đã ở rank 1–2 ngay lượt đầu (12/15 câu chỉ cần một lượt search). Lỗi bão hòa chủ yếu xảy ra với dense. Mức chênh 13 so với 14 nằm trong dao động giữa các lần chạy: agent tự viết query khác nhau giữa các lần (ví dụ q07, q04).
2. **Lỗi 2: evidence bị cắt sang chunk kề** (q07 ở hybrid, BM25 và Cải thiện 1; điểm 1). Câu trả lời đúng một phần ("models giảm độ phức tạp vô hạn"), nhưng thiếu ý chính "focus on relevant information rather than detailed representations". Chunk gold `b4376d` (ký tự 794–1493 trang 11) không nằm trong top-10 hybrid (dense xếp hạng 9), trong khi chunk ngay trước nó `153751` (ký tự 0–792 cùng trang) ở rank 2. Chunking Day 6 đã cắt đôi đoạn văn.
   - *Cách thử:* Cải thiện 2 (thêm chunk lân cận ±1, budget 5000 để chứa thêm). *Kết quả:* q07 lên điểm 2 và trích dẫn đúng `b4376d`; Context Evidence Rate của agent 15/15; grounded 2,00; citation 24/24; không có từ chối nhầm. Lợi ích phụ: budget lớn hơn cũng làm giảm lỗi 1.
3. **Lý do chọn:** tiêu chí được ghi **trước** khi chạy các cấu hình cải thiện và trước khi chạm test (`results/selection_criteria.md`). Thứ tự: 0 lỗi hệ thống → grounded trung bình → abstention/từ chối nhầm → citation hợp lệ → Context Evidence Rate → token/thời gian. Cải thiện 2 đứng đầu ở tiêu chí grounded (2,00 so với 1,93 của hybrid và Cải thiện 1), bằng điểm ở abstention và citation. Đánh đổi chấp nhận: tăng khoảng 42% token đầu vào (49,5k so với 35,0k trên dev), thời gian trả lời gần như không đổi. Trên test, cấu hình chọn đạt 5/5 câu điểm 2 so với 4/5 của baseline. Lỗi duy nhất của baseline trên test (q20) đúng là lỗi 1, nên kết quả test nhất quán với phân tích trên dev.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
    - **Tool** (function calling, `mode=ANY` nên mỗi lượt model bắt buộc gọi một tool): `search(query)` gọi retriever đang cấu hình; `answer(answer, citations)`; `abstain(reason)`. Model tự viết query; gold và `expected_answer` không bao giờ đi vào prompt hay retriever. `filter` section chỉ truyền khi câu hỏi có sẵn.
    - **Prompt và context:** mỗi lượt gọi model là stateless, gồm câu hỏi, danh sách query đã chạy và context tích lũy mới nhất, nên kích thước prompt bị chặn bởi budget chứ không tăng theo số lượt. Chunk trùng giữa các lượt bị loại. Chunk ID gốc dài 64 ký tự hex nên model thấy alias ngắn `c1, c2…`; sau đó code ánh xạ lại về ID thật trong câu trả lời, citation và trace. System prompt yêu cầu chỉ dùng context, coi nội dung tài liệu là dữ liệu chứ không phải chỉ dẫn, trích `[cN]` sau mỗi claim, và từ chối nếu thiếu bất kỳ phần nào của câu hỏi.
    - **Điều kiện dừng:** model gọi `answer` hoặc `abstain`; hết 3 lượt search thì chỉ còn được gọi `answer`/`abstain` (qua `allowed_function_names`); tối đa 4 quyết định mỗi câu. Câu trả lời có citation không nằm trong context cuối hoặc không xuất hiện dạng `[cN]` trong văn bản sẽ bị chuyển thành từ chối (event `invalid_or_missing_citations`). Kiểm tra này chỉ xác nhận ID; việc nguồn có hỗ trợ claim hay không được chấm tay.
    - **Xử lý lỗi:** lỗi model (429/5xx/mạng) được thử lại tối đa 4 lần với backoff lũy thừa. Nếu vẫn lỗi, hoặc vector store lỗi, câu đó được ghi `status: "error"` kèm thông báo, không có câu trả lời giả hay fallback. Trong tất cả các lần chạy: 0 lỗi model, 0 lỗi vector store, mọi lượt gọi thành công ngay lần đầu.

Số lượt gọi ghi theo dạng `số lượt search → số câu`, kèm tổng số lượt gọi LLM.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline dense / dev | 17/17 (100%) | 1→9, 2→1, 3→7; 49 LLM | 3/1/11; 1,53 | 13/14 (92,9%) | 12/12 | 2/2 (100%) | 3/15 (20,0%) |
| BM25 / dev | 17/17 (100%) | 1→12, 3→5; 44 LLM | 1/1/13; 1,80 | 17/18 (94,4%) | 14/14 | 2/2 (100%) | 1/15 (6,7%) |
| Hybrid / dev | 17/17 (100%) | 1→12, 2→1, 3→4; 43 LLM | 0/1/14; 1,93 | 20/20 (100%) | 15/15 | 2/2 (100%) | 0/15 (0%) |
| Cải thiện 1 (rank) / dev | 17/17 (100%) | 1→12, 2→1, 3→4; 43 LLM | 0/1/14; 1,93 | 19/19 (100%) | 15/15 | 2/2 (100%) | 0/15 (0%) |
| Cải thiện 2 (lân cận, đã chọn) / dev | 17/17 (100%) | 1→12, 2→1, 3→4; 43 LLM | 0/0/15; 2,00 | 24/24 (100%) | 15/15 | 2/2 (100%) | 0/15 (0%) |
| Cấu hình đã chọn / test | 7/7 (100%) | 1→5, 3→2; 18 LLM | 0/0/5; 2,00 | 7/7 (100%) | 5/5 | 2/2 (100%) | 0/5 (0%) |
| Baseline dense / test | 7/7 (100%) | 1→4, 3→3; 20 LLM | 1/0/4; 1,60 | 5/5 (100%) | 4/4 | 2/2 (100%) | 1/5 (20,0%) |

Chấm context trên 5 câu test có đáp án (0 = không có evidence, 1 = có nhưng thiếu điều kiện/nguồn, 2 = đủ evidence và nguồn): **baseline** 2 điểm: 4 câu (q16–q19), 0 điểm: 1 câu (q20); **cấu hình chọn** 2 điểm: 5 câu. Lý do từng câu ghi trong `reviews.json` (trường `context_grade`).

Hai citation không hợp lệ trên dev: dense q07 trích chunk trang 11 (`35daee`) cho claim "dùng model vì bị ràng buộc bởi model of computation", nhưng chunk không nói đó là lý do; BM25 q01 trích chunk trang 2 (nói về *ngôn ngữ lập trình*) cho claim về *ngôn ngữ tự nhiên*.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q20 (test, cấu hình chọn) | Có đáp án | 1 search `longest-paths traveling-salesperson reduction` → 3 hit (p38, p39, p46) + 6 chunk lân cận; chunk gold `1525e7` (p39) ở rank 2, nằm trong context 5000 ký tự | "LPT và TS đều reduce về Hamiltonian paths (HP); HP reduce về vertex cover (VC), VC reduce về 3-SAT" `[1525e7…]` | Grounded 2, citation 1/1 hỗ trợ, context 2: đủ cả chuỗi reduction, đúng nguồn và trang |
| q_test_unabs_01 (test, cấu hình chọn) | Thiếu evidence | 3 search: `halting problem undecidable`, `halting`, `undecidable undecidability 3.6` → khoảng 25 chunk (p10, p38, p17, p53, p47…), không chunk nào nhắc halting | Từ chối; lý do: "context không có thông tin về halting problem hay chứng minh tính không quyết định được" | Abstention đúng: tìm nhiều cách diễn đạt rồi mới từ chối, không thêm khẳng định |
| q20 (test, baseline dense) | Có đáp án, **lỗi** | 3 search; chunk gold p39 ở rank 3 trong cả 3 lượt nhưng hai chunk p38 đã chiếm hết 2500 ký tự | Từ chối nhầm | Grounded 0, context 0. Đúng lỗi 1 (context bão hòa) đã phân tích trên dev; cấu hình chọn trả lời đúng câu này |
| q12 (dev, baseline dense) | Có đáp án, **lỗi** | Lượt 3 lấy chunk gold p21 ở rank 1 nhưng context đã đầy từ lượt 1–2 | Từ chối nhầm | Grounded 0. Động lực cho Cải thiện 1 |
| q07 (dev, hybrid) | Có đáp án, **lỗi** | 1 search `models detail real world` → p10, p11 (`153751`), p43; chunk gold `b4376d` kề sau `153751` nhưng không được lấy | Đúng một phần, `[955a69…][153751…]` | Grounded 1, citation 2/2 hỗ trợ nhưng thiếu ý chính. Cải thiện 2 sửa thành điểm 2 |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
    - Dữ liệu: `data/chunks.json`, `data/questions.json`, `data/agent_cases.json`; tạo lại bằng `python scripts/prepare_data.py --day6-benchmark <repo-day6>/starter/data/dsa_benchmark.json`.
    - Retrieval: `python scripts/benchmark_retrieval.py --split dev` (và `--split test` sau khi chốt) → `results/retrieval/summary_{dev,test}.json`, `traces_{dev,test}.json` (rank, score, nguồn, trang, section, hit, context từng budget). Baseline starter: `results/baseline-tfidf-dev/`.
    - Agent: `python scripts/run_agent.py --split dev --retriever dense` (baseline); `--retriever bm25`; `--retriever hybrid`; `--retriever hybrid --context-order rank`; `--retriever hybrid_nb --budget 5000`. Mỗi lần chạy tạo `results/agent/<cấu hình>-<split>/` gồm `config.json`, `traces.json` (mọi lượt gọi LLM kèm token và thời gian, mọi search kèm query/filter/kết quả/rank/score, context cuối, câu trả lời, citation, lỗi), `run_summary.json`, `review_sheet.json`, `reviews.json` (nhãn chấm), `metrics.json` (từ `starter/agent_metrics.py`).
    - In phiếu chấm: `python scripts/show_review.py results/agent/<cấu hình>-<split>`. Tổng hợp nhãn: `python starter/agent_metrics.py --cases data/agent_cases.json --traces <dir>/traces.json --reviews <dir>/reviews.json --split <split> --output <dir>/metrics.json`.
    - Test: `python -m unittest discover -s tests -v` (20 test, trên Windows cần `PYTHONUTF8=1`).
    - Thời gian và chi phí: xem hai bảng trên. Toàn bộ thử nghiệm agent (5 cấu hình dev, 2 cấu hình test, 99 câu) dùng 234,9k token đầu vào và 45,1k token đầu ra + thinking.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
    - **Trước/sau trên dev:** từ baseline dense sang cấu hình chọn, grounded trung bình 1,53 → 2,00, từ chối nhầm 3/15 → 0/15, Context Evidence Rate của agent 10/15 → 15/15, citation hợp lệ 13/14 → 24/24. Trên test: 1,60 → 2,00, từ chối nhầm 1/5 → 0/5. Cả hai cấu hình đều từ chối đúng 4/4 câu thiếu evidence.
    - **Đánh đổi:** cấu hình chọn tốn khoảng 42% token đầu vào hơn hybrid thường vì context gấp đôi (trung bình khoảng 4,7k so với 2,1k ký tự). Thời gian trả lời gần như không đổi (6,4 so với 7,0 s/câu trên dev), vì phần lớn thời gian là gọi model. Retrieval một lượt vẫn dưới 40 ms. Cải thiện 1 không tốn thêm chi phí nhưng chưa cho thấy lợi ích về chất lượng trên bộ dev này.
    - **Giới hạn:**
        - Benchmark nhỏ (15 + 5 câu có đáp án), một lần chạy cho mỗi cấu hình. Chênh lệch 1 câu (6,7 điểm % trên dev, 20 điểm % trên test) có thể nằm trong dao động do agent tự viết query khác nhau giữa các lần chạy, dù `temperature=0`.
        - Nhãn chấm do một người chấm, có hỗ trợ của trợ lý AI khi đọc đối chiếu chunk; người làm bài chịu trách nhiệm rà soát lại nhãn trong các file `reviews.json`.
        - Câu hỏi chủ yếu cần một đoạn evidence, chưa có câu multi-hop thực sự. Câu thiếu evidence được kiểm tra bằng từ khóa, nên có thể sót cách diễn đạt khác.
        - Chưa quy đổi chi phí sang USD.
        - Sau khi chốt, retrieval trên test được chạy cho cả 5 phương pháp để báo cáo; kết quả này không được dùng để chỉnh cấu hình.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện thử thách nâng cao.
