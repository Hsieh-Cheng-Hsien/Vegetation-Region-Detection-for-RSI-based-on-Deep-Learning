	8gD鞹@8gD鞹@!8gD鞹@	??????B???????B?!??????B?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$8gD鞹@"??u????A A?c???@YvOjM??*	23333X?@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???x??@!?R?U??W@)]?C??k@1?Ʒn??W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!?W<"@)??????1?{<?@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??H?}??!??E?s???)?A`??"??1?[0?"??:Preprocessing2F
Iterator::Model?g??s???!?|?p????)?ZӼ???1H??XǦ??:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::ConcatenateM?O???!??{??)?j+??݃?1T;nF???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?j+??݃?!T;nF???)???Q?~?1܈ox???:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1?|?!? ?U??)y?&1?|?1? ?U??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::PrefetchǺ???v?!?f6B????)Ǻ???v?1?f6B????:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenatea??+ey?!;??????)??_vOv?1|u??V:??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipH?}8g??!&?8???@)?J?4q?1M?Q??̺?:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!?_Y?Y??)HP?s?b?1?_Y?Y??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!}U?7??)/n??b?1}U?7??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensor????Mb@?!??M?Յ??)????Mb@?1??M?Յ??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensor-C??6:?!l,>pk??)-C??6:?1l,>pk??:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlicea2U0*?3?!?B](??~?)a2U0*?3?1?B](??~?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??????B?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	"??u????"??u????!"??u????      ??!       "      ??!       *      ??!       2	 A?c???@ A?c???@! A?c???@:      ??!       B      ??!       J	vOjM??vOjM??!vOjM??R      ??!       Z	vOjM??vOjM??!vOjM??JCPU_ONLYY??????B?b 