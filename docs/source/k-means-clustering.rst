k-Means Clustering
==================

Vài dòng giới thiệu về thuật toán
---------------------------------

| k-Means Clustering là một trong những thuật toán đơn giản và nổi tiếng
  nhất của làng Machine Learning. Nó là phương pháp tuy đơn giản nhưng
  đặc biệt hiệu quả trong bài toán Unsupervised Learning, khi mà dữ liệu
  của bạn chưa được phân loại (unlabeled). Mục tiêu của thuật toán này
  là chia nhỏ data của bạn thành k groups dựa trên features được cung
  cấp.. Các điểm dữ liệu được xếp vào trong từng nhóm dựa trên sự giống
  nhau về đặc điểm nhận dạng .

Ứng dụng trong thực tế
----------------------

| Vì khả năng cơ bản của k-Means Clustering là chia nhỏ dữ liệu ban đầu
  thành các nhóm nhỏ, tất cả hoạt động dựa trên thuật toán mà không yêu
  cầu bất kì kiến thức của người sử dụng về dữ liệu đã được thu thập(to
  - nhỏ, xấu - đẹp, méo - tròn). Nó có thể được sử dụng để xác nhận các
  giả thiết về việc nên phân chia làm bao nhiêu nhóm, là những nhóm nào,
  khi mà lượng dữ liệu thu được lớn và phức tạp. Khi mà 2 thông số trên
  được xác định, bất kì một sample mới sẽ đễ dàng được gán nhãn vào vị
  trí chính xác.
| Đây là một thuật toán linh hoạt có thể được ứng dụng vào bất kì quy
  trình **phân loại và chia nhóm**. Một vài ví dụ có thể kể đến sau đây:

#. Trong giao dich ngân hàng, việc phân loại dữ liệu khách hàng là đặc
   biệt quan trọng, thường được dựa vào đó để đưa ra các chính sách
   chung cho toàn hệ thống hay là có những chính sách chăm sóc đến từng
   khách hàng. Một vài cách phân loại dựa trên hành vi người dùng như
   sau:

   -  Phân loại dựa trên lịch sử thanh toán (chi tiêu)

   -  Phân loại dựa trên hoạt động trên ứng dụng di động, trên website,
      hay trên nền tảng ATM

   -  Định nghĩa tính cách cá nhân khách hàng dựa trên mối quan tâm của
      họ (thông qua lịch sử mua sắm)

   -  Tạo profile của khách hàng dựa trên dữ liệu theo dõi hoạt động

#. Phân loại sáng chế của cục sở hữu khoa học kĩ thuật:

   -  Nhóm các sáng chế dựa trên hoạt động kinh doanh

   -  Nhóm các sáng chế dựa trên khối ngành sản suất

#. Phân loại cảm biến theo chức năng:

   -  Phương pháp phát hiện hoạt động trong nhóm cảm biến chuyển động

   -  Chia nhóm ảnh

   -  Phân loại file audio

   -  Chia nhóm trong theo dõi sức khỏe

Đặc biệt, việc theo dõi sự thay đổi của những điểm dữ liệu bị theo dõi
(một số cá nhân trong dữ liệu khách hàng của ngân hàng) có thể được sử
dụng để theo dõi xu hướng hoạt động. Điều này thường được sử dụng nhiều
trong theo dõi hoạt động mua sắm, góp phần giúp cho các tập đoàn lớn như
Nike, Zara, Uniqlo... nắm bắt được người dùng và đưa ra những chính sách
góp phần nâng cao doanh số cũng như thúc đẩy hoạt động mua sắm.

Lí thuyết
---------

| Bắt đầu bằng một ví dụ đơn giản, trong một lớp có 50 học sinh, chúng
  ta cần chia thành 3 nhóm dựa trên chiều cao của các cá nhân trong lớp.
  Có thể tạm gọi là "Cao", "Trung Bình", và "Thấp". Việc dán nhãn như
  trên chỉ mang tính minh họa và hoàn toàn không ảnh hưởng đến kết quả
  của bài toán. Như vậy chúng ta có một tập dữ liệu
  :math:`\mathbf{x}_i, i=1 \cdots 50` là 50 giá trị chiều cao của các
  thành viên trong lớp.
| Như đã nói ở trên, chúng ta muốn chia lớp thành 3 nhóm, do đó chúng ta
  có :math:`\mathbf{c}_j, j = 1 \cdots 3` là kí hiệu 3 điểm trung tâm
  của mỗi nhóm. :math:`\mathbf{c}_j` là một giá trị về chiều cao (mét)
  và những cá nhân trong lớp có chiều cao gần với :math:`\mathbf{c}_j`
  sẽ được xếp vào nhóm :math:`j`. Tôi để các giá trị
  :math:`\mathbf{c}_j`, :math:`\mathbf{x}_i` in đậm thể hiện đó là
  vector vì trong ví dụ này, chúng ta chỉ có một tham số được đo lường
  đó là chiều cao, nhưng để không mất tính tổng quát,
  :math:`\mathbf{c}_j` và :math:`\mathbf{x}_i` có thể chứa thêm nhiều
  tham số khác, như ngày tháng nắm sinh, cân nặng, ...
|  Việc tiếp theo là định nghĩa khoảng cách về chiều cao giữa một thành
  viên bất kì :math:`\mathbf{x}_i` đến điểm trung tâm của nhóm
  :math:`\mathbf{c}_j`. Chúng ta có định nghĩa:

.. math:: d_j\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right) = \left\| {{{\bf{x}}_i} - {{\bf{c}}_j}} \right\|_2^2

 là hàm khoảng cách giữa hai điểm này.

-  :math:`d_j\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right)` là kí hiệu
   khoảng cách (lấy chữ cái đầu trong "distance") giữa hai điểm
   :math:`\mathbf{c}_i` và :math:`\mathbf{x}_i`.

-  :math:`\left\| {\cdots} \right\|_2^2` là bình phương của hàm
   :math:`L^2-norm` (hay còn được gọi là vector norm)

| Giải thích một chút về hàm :math:`L^2-norm` này. Công thức toán học
  của nó như sau:

  .. math:: {\left\| {{{\bf{x}}_i} - {{\bf{c}}_i}} \right\|_2} = \sqrt {\sum\limits_{k = 1}^n {{{\left| {{x_k} - {c_k}} \right|}^2}} }

   với vector
  :math:`{{\bf{x}}_i} = \left\langle {{x_1},{x_2},...,{x_k},...,{x_n}} \right\rangle`
  và
  :math:`{{\bf{c}}_i} = \left\langle {{c_1},{c_2},...,{c_k},...,{c_n}} \right\rangle`.
| Trong ví dụ này, :math:`\mathbf{c}_j` và :math:`\mathbf{x}_i` chỉ có 1
  tham số duy nhất nên giả sử ta có: :math:`\mathbf{x}_{34} = 1.32m` thể
  hiện thành viên thứ 34 trong lớp có chiều cao :math:`1.32m`, và
  :math:`\mathbf{c}_2 = 1.30m` thể hiện nhóm :math:`j=2` có giá trị
  trung tâm là :math:`1.30m`, chúng ta có thể tính ra được giá trị của
  hàm khoảng cách:

  .. math:: d_j\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right) = {\left( {\sqrt {{{\left( {1.32 - 1.30} \right)}^2}} } \right)^2} = {\left( {0.02} \right)^2} = 0.004

   (Các bạn đừng quên là chúng ta đang tính bình phương nhé).
| Chúng ta có 3 giá trị của 3 nhóm khi :math:`j=1 \cdots 3`, do đó hàm
  tổng khoảng cách từ 1 điểm :math:`\mathbf{x}_i` đến cả 3 điểm
  :math:`\mathbf{c}_i` là:

  .. math:: \sum\limits_{j = 1}^k {d\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right)}  = \sum\limits_{j = 1}^k {\left\| {{{\bf{x}}_i} - {{\bf{c}}_j}} \right\|_2^2}

   ở đây :math:`k=3` là số clusters, như đã được định nghĩa ở trên.
| Đến đây, chúng ta đã tính được khoảng cách từ 1 điểm
  :math:`\mathbf{x}_i`, nhưng vì chúng ta có tất cả 50 thành viên trong
  lớp, do đó, hàm khoảng cách cần phải nâng cấp lên 1 lần nữa.

  .. math:: \sum\limits_{i = 1}^m {\sum\limits_{j = 1}^k {d\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right)} }  = \sum\limits_{i = 1}^m {\sum\limits_{j = 1}^k {\left\| {{{\bf{x}}_i} - {{\bf{c}}_j}} \right\|_2^2} }

   :math:`m=50` là số sample trong data hay số thành viên trong lớp.

| Đến đây, chúng ta đã có một hàm có thể ước lượng được ưu nhược điểm
  của :math:`\mathbf{c}_j`. Càng chọn :math:`\mathbf{c}_j` sao cho hàm
  trên có giá trị càng nhỏ, thuật toán càng tốt hơn. Khi mà giá trị của
  :math:`\mathbf{x}_i` là cố định thì **PHƯƠNG PHÁP CHỌN**
  :math:`\mathbf{c}_j` là trung tâm của thuật toán k-Means Clustering.
| Bây giờ, ta có thể đặt:

  .. math:: J = \sum\limits_{i = 1}^m {\sum\limits_{j = 1}^k {d\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right)} }  = \sum\limits_{i = 1}^m {\sum\limits_{j = 1}^k {\left\| {{{\bf{x}}_i} - {{\bf{c}}_j}} \right\|_2^2} }

   là hàm mục tiêu (objective function) hay hàm phí tổn (cost function)
  cho bài toán.
| Và thuật toán k-Means Clustering đi tìm giá trị cực tiểu:

  .. math:: {J_{\min }} = \arg \mathop {\min }\limits_c \sum\limits_{i = 1}^m {\sum\limits_{j = 1}^k {d\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right)} }

Thuật toán
----------

Bây giờ là lúc chúng ta cùng nhau tìm hiểu về quy luật để từ những giá
trị :math:`\mathbf{c}_j` được chọn ngẫu nhiên, chúng ta có thể đi đến
giá trị mong muốn đảm bảo việc phân nhóm thành công. Sơ đồ thuật toán
như sau:

#. Input: :math:`k`- số cluster, :math:`\mathbf{x}_i, i=1 \cdots 50`-
   data

#. Khởi tạo: Đặt :math:`\mathbf{c}_j, j = 1 \cdots 3` tại các vị trí bất
   kì

#. Lặp lại đến khi hội tụ:

   -  Với mỗi điểm :math:`\mathbf{x}_i`:

      -  Tìm điểm trung tâm gần nhất :math:`\mathbf{c}_j` của nó (so
         sánh :math:`d_j\left( {{{\bf{x}}_i},{{\bf{c}}_j}} \right)`

      -  Xếp :math:`\mathbf{x}_i` vào Cluser :math:`j`

   -  Với mỗi Cluster :math:`j=1 \cdots k`:

      -  | Điểm trung tâm mới :math:`\mathbf{c}_{j}-new` là trung bình
           cộng của tất cả các điểm :math:`\mathbf{x}_i` đã được xếp vào
           trong Cluster :math:`j` từ bước trên.
         | 

           .. math:: {{\bf{c}}_j} = \frac{1}{{\left| {{S_j}} \right|}}\sum\limits_{{{\bf{x}}_i} \in {S_j}} {{{\bf{x}}_i}}

            :math:`S_j` là tập các điểm dữ liệu :math:`\mathbf{x}_i` của
           cluster thứ :math:`j` trong tập dữ liệu.

#. Kết thúc khi không có sự thay đổi từ cluster này sang cluster kia.

| Dựa vào công thức tính toán của hàm, chúng ta có thể đặt ra câu hỏi
  là: **Kiểu dữ liệu nào thì thích hợp áp dụng k-Means Clustering?**.
  Kiểu dữ liệu số học(numerical) hay kiểu dữ liệu phân loại
  (categorical)?
| Câu trả lời là chỉ có kiểu dữ liệu số học được chấp nhận mà thôi. Bạn
  thấy đấy, tất cả các bước kể trên, chúng ta đi tính khoảng cách từ 1
  điểm trong tập dữ liệu :math:`\mathbf{x}_i` đến một điểm trung tâm
  :math:`\mathbf{c}_j`, đây là phương pháp tính khoảng cách hình học
  Euclid chúng ta đã học từ bậc phổ thông. Đối với các tập dữ liệu chứa
  các thuộc tính về Categorical, **không áp dụng k-Means Clustering**
  được nhé!
| Trang web
  `này <http://stanford.edu/class/ee103/visualizations/kmeans/kmeans.html>`__
  cho phép bạn nhìn thấy sự thay đổi một cách trực quan thuật toán. Bạn
  nên ghé qua để có thể hiểu hơn về thuật toán.

Lựa chọn K
----------

| Thuật toán k-Means yêu cầu một điều kiện tiên quyết đó là phải biết
  trước chính xác giá trị của k. Đây là một bước đòi hỏi kĩ năng và kinh
  nghiệm của người phân tích dữ liệu. **Chọn k như thế nào là phù
  hợp?**. Để tìm số lượng cluster trong dữ liệu, người dùng phải chạy
  thuật toán với một khoảng giá trị của k và so sánh kết quả thu được.
  Cơ bản, không có phương pháp xác định giá trị chính xác nào của k,
  nhưng một giá trị ước lượng đúng có thể thu được bằng cách sử dụng
  phương pháp sau:
| Một đại lượng thường được sử dụng để so sánh kết quả giữa các giá trị
  khác nhau của k là khoảng cách trung bình của các điểm dữ liệu
  :math:`\mathbf{x}_i` tới tâm của cluster :math:`\mathbf{c}_j`. Vì việc
  tiếp tục tăng giá trị của k luôn làm giảm khoảng cách trung bình này,
  và trường hợp cực trị đó là giá trị khoảng cách tiến tới 0 khi
  :math:`k=m` (:math:`m` là số lượng điểm dữ liệu trong data). Do đó,
  chúng ta không dùng đại lượng này một cách trực tiếp. Thay vào đó,
  chúng ta sẽ dùng đồ thị về khoảng cách trung bình của các điểm là hàm
  của k và xác định điểm gấp khúc "elbow point", khi mà độ dốc của đồ
  thị thay đổi đột ngột, có thể được sử dụng để xác định k.
| Một số các phương pháp khác tồn tại để xác định giá trị thích hợp của
  k, đó là cross-validation, information criteria, the information
  theoretic jump method, the silhouette method, và the G-means algorithm
  (Xin được để nguyên văn để độc giả tiện tra cứu). Hơn nữa, việc quan
  sát sự phân bố của dữ liệu trong các nhóm cung cấp cái nhìn vào việc
  làm thế nào thuật toán phân chia dữ liệu đối với mỗi giá trị của k.
  Việc chọn một số điểm dữ liệu ngẫu nhiên và vẽ đồ thị là cần thiết
  trong đa số trường hợp.