?	xz?,S??@xz?,S??@!xz?,S??@	"5F??F?"5F??F?!"5F??F?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$xz?,S??@??/?$??A`??"??@Y9??m4???*	effff?@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?S㥛D@!96Z??W@)??ǘ?@1??9F?W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(\?????!5???u?@)ꕲq???1Vz1@??@:Preprocessing2F
Iterator::Model?<,Ԛ???!R#q$cS??)??A?f??1?+?Fi??:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatǺ?????!??%~,???)??A?f??1?+?Fi??:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenatea2U0*???!??@?p???)HP?sׂ?1d?=?6???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???S????!????@)?5?;Nс?1ˏ?C????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?(??0??!?e?J\~??)"??u????1z???_???:Preprocessing2U
Iterator::Model::ParallelMapV2?St$????!???:t???)?St$????1???:t???:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate?&S???!A?L?R??)????Mb??1??5?????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?+e?Xw?!6^f {???)?+e?Xw?16^f {???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???_vOn?!?ϱ?????)???_vOn?1?ϱ?????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?X?!,c(?f???)?~j?t?X?1,c(?f???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensorǺ???F?!??%~,???)Ǻ???F?1??%~,???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensor-C??6:?!r+??S??)-C??6:?1r+??S??:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlice-C??6:?!r+??S??)-C??6:?1r+??S??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9#5F??F?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??/?$????/?$??!??/?$??      ??!       "      ??!       *      ??!       2	`??"??@`??"??@!`??"??@:      ??!       B      ??!       J	9??m4???9??m4???!9??m4???R      ??!       Z	9??m4???9??m4???!9??m4???JCPU_ONLYY#5F??F?b Y      Y@q{.?????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 