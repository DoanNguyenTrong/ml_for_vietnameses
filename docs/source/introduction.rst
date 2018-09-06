Giới thiệu về Machine Learning
==============================

Giới thiệu chung
----------------

**Định nghĩa**:
    **Machine Learning(ML)** là một nhánh của ngành khoa học máy tính (Computer Science) tập trung sử dụng các phương pháp thống kê
    **(Statistical techniques)** để trao cho các hệ thống tính toán (máy tính) khả năng "học hỏi" từ tập dữ liệu (data) mà không
    cần phải được lập trình một cách chuyên biệt.

Theo định nghĩa trên bạn nên chú ý rằng khả năng "học hỏi" của các thuật toán ML dựa trên một loạt các phép toán **thống kê xác suất**
và các quyết định/dự đoán do thuật toán đưa ra hoàn toàn phụ thuộc vào **dữ liệu (data)** đã thu thập được.
Chính vì thế data là một thứ cốt lõi và cực kì quan trọng trong ML. 

Có khá nhiều nhánh công việc giải quyết các bài toán
xung quanh data và ML cũng chỉ là 1 trong số chúng mà thôi.

.. image:: /images/Data-Science-Skills-Udacity-Matrix.png
    :width: 90%
    :align: center

`nguồn  <https://1onjea25cyhx3uvxgs4vu325-wpengine.netdna-ssl.com/wp-content/uploads/2014/11/Data-Science-Skills-Udacity-Matrix.png>`_

Hình trên cho thấy các kĩ năng cần thiết nếu bạn chọn một trong 4 công việc liên quan đến data.
Trong đó kiến thức về ML là cực kì quan trọng đối với kĩ sư ML (Machine Learning Engineer) và 
nhà khoa học dữ liệu (Data Scientist).


.. note:: Nguồn tài liệu tiếng Việt là không đủ và không cập nhật nếu bạn muốn tiến xa hơn trong sự nghiệp của mình. Nếu bạn muốn nâng level lên, mình
    khuyên thật lòng là bạn nên cân nhắc về việc dành thời gian trau dồi ngoại ngữ của mình. Cũng vì lẽ đó các thuật ngữ
    được để nguyên tiếng Anh hoặc chú thích với mục đích khiến bạn không bị bỡ ngỡ. Mình đã phải rất chật vật với các định nghĩa toán học
    đã được Việt hóa khi học bằng tiếng Anh (vd: dạng toàn phương). Thế nên là mình tin rằng việc giữ nguyên hay chú thích sẽ giúp bạn về lâu về dài.


ML tasks (Các tác vụ của ML)
----------------------------

Có 2 tác vụ chính của ML phụ thuộc vào việc dữ liệu đã được tiền xử lí
(phân loại) hay chưa đó là **Supervised Learning** và **Unsupervised Learning**.

Supervised Learning
-------------------

Trong đó các thuật toán của DSSupervised Learning **train** mô hình(**model**) dựa trên
data được dán nhãn (**labeled**) một cách *thủ công*. **model** được thuật toán sinh ra
thể hiện mối quan hệ giữa thông số của đầu vào (**input**) và đầu ra (**output**). Sau đó
model này sẽ được thẩm tra xác minh (**testing**) độ chính xác bằng cách dùng nó để dự đoán giá trị output
mong muốn. Độ chính xác của thuật toán được thẩm tra bằng cách so sánh giá trị output dự đoán bởi
thuật toán và giá trị đúng đã gắn nhãn trước đó. Chú ý là data dùng cho training và testing phải không 
bị trùng lặp. Thường thì data được gom lại thành một tệp, thuật toán sẽ thực hiện việc chia data một cách
ngẫu nhiên thành 2 phần để phục vụ cho training và testing. 

VD: chúng ta có một tập dữ liệu là các bức ảnh **có** và **không có** mặt người. Thuật toán phân loại (classification - bạn sẽ được tìm hiểu sau đó)
sẽ tìm ra một model thể hiện mối liên hệ dữa thông tin của bức ảnh và label của nó. Sau đó, bạn có thể nhập vào một bức ảnh bất kì và model này sẽ chỉ
ra trong bức ảnh đó là hình của một khuôn mặt hay không.

..note: Các nhà xây dựng thuật toán đều cố gắng đưa ra các đặc điểm (**feature**) tiêu biểu thể hiện rõ ràng mối liên hệ giữa training data và
label của nó. Nhưng vì đặc trưng là tập dữ liệu chỉ chiếm một phần nhỏ trong tổng số các khả năng có thể xảy ra và các đặc điểm đó không phải là
tất cả các đặc điểm của model. Do đó **xác suất dự đoán của mọi thuật toán không bao giờ có thể chính xác 100%**. Kể cả con người cũng vẫn có thể mắc sai lầm.
Hiện nay có một vài thuật toán có thể cho ra kết quả cao hơn xác suất dự đoán chính xác của con người nhưng còn quá sớm để nói là tương lai các hệ thống
trí tuệ nhân tạo sẽ thống trị thế giới. Các bức tranh về Terminator, Transformer, ... vẫn còn ở rất xa so với nhân loại năm 2018.

Cụ thể hơn, trong Supervised learning gồm:

* **Semi-Supervised learning**: thuật toán giải quyết các bài toán với data chưa hoàn thiện. Thường thì training set chứa rất nhiều instance chưa được gán label. Do đó thuật toán này phải đưa ra dự đoán label cho các instance này.

* **Active learning**: thuật toán được cung cấp label của một tập nhỏ instance và nhiệm vụ của nó là lựa chọn label cho các đối tượng một cách tối ưu. Thuật toán này thường được dùng để gán label cho instance.

* **Reinforecement Learning**: là phương pháp các agent (trợ lí) có nhiệm vụ hành động trong một *môi trường* với mục tiêu tối ưu hóa các giải thưởng. Có thể hiểu một cách đơn giản là nếu một cá nhân cư xử tốt thì sẽ được phần thường, ngược lại sẽ bị trừng phạt nếu cư xử không tốt. Các thuật toán reinforecement learning cũng có những cơ chế thưởng phạt và tìm cách đạt được càng nhiều phần thưởng càng tốt. Đây là một phương pháp rất thú vị được ứng dụng trong rất nhiều lĩnh vực khác nhau: game theory, control theory, operations research, information theory, simulation-based optimization, multi-agent systems, swarm intelligence, statistics and genetic algorithms.

Unsupervised Learning
---------------------

Trong Unsupervised learning, dữ liệu chưa được gán label, do đó thuật toán phải tìm ra các điểm chung tiêu biểu trong data. Mục tiêu là tìm ra
các kiểu mẫu  tiêu biểu tiềm ẩn trong tập dữ liệu (dataset) phục vụ cho bài toán phân loại, chia nhóm (classification, clustering).

Unsupervised Learning thường được dùng trong các dữ liệu về giao dịch. Bạn có thể có một tập rất lớn dữ liệu khách hàng và các giao dịch, mua bán của họ.
Nhưng tập dữ liệu này là phức tạp mà bạn khó có thể tìm ra sự tương tự giữa thông tin cá nhân của khách hàng và loại hàng họ mua. Thuật toán có Unsupervised
learning có thể phát hiện là phụ nữ trong một khoảng độ tuổi, những người mua xà phòng không mùi sẽ có xác suất rất cao là đang mang thai. Do đó các chiến dịch
marketing cho nhóm đối tượng phụ nữ đang mang thai và trẻ nhỏ có thể hướng đến những người dùng này. Do đó họ có thể giảm thiểu được chi phí thay vì quảng cáo các
sản phẩm này tới trẻ vị thành niên hay người già và đem lại hiệu quả cao hơn.

Unsupervised Learning tìm kiếm trong đống dữ liệu rối rắm và dường như không mấy liên quan và sắp xếp lại theo các cách có thể có ý nghĩa.
Unsupervised Learning thường được dùng trong việc phát hiện các hành vi bất thường như các gian lận trong thanh toán bằng thẻ ghi nợ (credit card) hay
đưa ra các gợi ý loại hàng hóa nào nên mua.

Các ứng dụng của Machine Learning
---------------------------------

*Một phương pháp khác phân loại các thuật toán ML dựa vào output của nó*:

* **Classification**: dữ liệu input được chia thành 2 hay nhiều class (lớp), và thuật toán có nhiệm vụ xếp các dữ liệu mới vào một hay nhiều class trong số các class trên. Đây thường là một bài toán Supervised learning. Bộ lọc email spam (spam filtering) là một ví dụ của classification, với input thường là email hay tin nhắn được gán label là "spam" và "not spam".

* **Regression**: đây cũng là một bài toán Supervised, với kết quả output thường là hàm liên tục thay vì các class rời rạc như trong classification.

* **Clustering**: một tập input được chia thành các nhóm nhỏ dựa vào các đặc trưng tương tự giữa các instance. Do đó đây là một bài toán Unsupervised learning.

* **Density estimation**: tìm sự phân bổ của inputs trong một số miền không gian.

* **Dimensional reduction** đơn giản hóa inputs bằng cách ép (mapping) chúng vào một miền không gian ít chiều hơn (lower-Dimensional space).

..note: Tóm lại note này giúp bạn có cái nhìn cơ bản về các thuật ngữ của ML. Có thể bạn sẽ bị bỡ ngỡ đôi chút về các diễn giải nếu là người mới bắt đầu tìm hiểu ML. Đừng lo, bạn sẽ
hoàn toàn nắm được những nội dung của note khi đi vào các bài sau.

Reference:
----------

#. https://en.wikipedia.org/wiki/Machine_learning
#. https://www.digitalocean.com/community/tutorials/an-introduction-to-machine-learning