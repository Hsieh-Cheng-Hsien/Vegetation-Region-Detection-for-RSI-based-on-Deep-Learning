?	aTR' ??@aTR' ??@!aTR' ??@	\???9?D?\???9?D?!\???9?D?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$aTR' ??@??x?&1??A?'???@Y??g??s??*	????.?@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map	?^)@!?{P?3EX@)]m????@1?k??4X@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?b?=y??!?sPþY@)_?L???1?Vq?????:Preprocessing2F
Iterator::Model?]K?=??!?%??<???)?N@aÓ?1??g?????:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?W[?????!?y?I??)lxz?,C??1?2?1V???:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::ConcatenateHP?sׂ?!?!?v9???)?J?4??1V@??w??:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate?j+??݃?!?.??R???)?J?4??1V@??w??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ?o_Ή?!???߲-??)vq?-??1]3v?]
??:Preprocessing2U
Iterator::Model::ParallelMapV2?<,Ԛ?}?!d?h~??)?<,Ԛ?}?1d?h~??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?HP?x?!?o.??P??)?HP?x?1?o.??P??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!Q'?[??@)?g??s?u?1???VMݶ?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!HZE?F??)U???N@s?1HZE?F??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_?LU?!9t???n??)??_?LU?19t???n??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensor????MbP?!??P$?A??)????MbP?1??P$?A??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensor-C??6J?!?Nm#???)-C??6J?1?Nm#???:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlicea2U0*?3?!?????t?)a2U0*?3?1?????t?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9]???9?D?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??x?&1????x?&1??!??x?&1??      ??!       "      ??!       *      ??!       2	?'???@?'???@!?'???@:      ??!       B      ??!       J	??g??s????g??s??!??g??s??R      ??!       Z	??g??s????g??s??!??g??s??JCPU_ONLYY]???9?D?b Y      Y@qcP??8?y?"?
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