	??&????@??&????@!??&????@	????v?????v?!????v?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??&????@l	??g???AP??n2??@Y??V?/???*	?????@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapQ?|?@!j?pt?V@)?o_?Y@1?r?VvV@:Preprocessing2F
Iterator::Model???Q???!O?|?@)?c?ZB??1??HE??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa??+e??!333333@)鷯???1?gL?????:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?Q?????!?O?|???)io???T??1?겹!??:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::ConcatenateM?O???!r??1????);?O??n??1?`Q?(X??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/?$???!??^x/???)??ǘ????1????WO??:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenatevq?-??!?ԭ.????)ŏ1w-!?1?Q?9 .??:Preprocessing2U
Iterator::Model::ParallelMapV2??H?}}?!OMt??F??)??H?}}?1OMt??F??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch{?G?zt?!`kZ?ך??){?G?zt?1`kZ?ך??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"lxz?,??!?t!??)@)_?Q?k?1?Z???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!g??^???)a2U0*?c?1g??^???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!?b?S?ˤ?)HP?s?b?1?b?S?ˤ?:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensor??H?}M?!OMt??F??)??H?}M?1OMt??F??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensora2U0*?3?!g??^?u?)a2U0*?3?1g??^?u?:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlice-C??6*?!q?@?(?l?)-C??6*?1q?@?(?l?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????v?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l	??g???l	??g???!l	??g???      ??!       "      ??!       *      ??!       2	P??n2??@P??n2??@!P??n2??@:      ??!       B      ??!       J	??V?/?????V?/???!??V?/???R      ??!       Z	??V?/?????V?/???!??V?/???JCPU_ONLYY????v?b 