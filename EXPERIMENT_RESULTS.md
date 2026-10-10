## 2026-10-09 V10修正版五轮/七轮既有轨迹CPU只读对比完成

精确绑定PathVQA44/42两次运行：五轮 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548`（commit4838875e385650d121319599fe776a1f5215733e），七轮 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_7ep_seed44_20261009_151030_637257`（b2724f30e18f8b559fa97dd6550677953a87226c），服务器均pathvqa/outputs/v10_head_fixed/下。仅复制既有train.log/report/state/diagnostics及六份Validation summary，Windows NumPy/Pillow计算，无模型/GPU操作、无服务器写入。两报告仅实验名/预算/保存轮数/commit/训练汇总不同；1691043参数、44/42、batch2/accum16、TF5/Accelerate1.12、归一化False、初始化审计相同；未保存数据hash/逐batch顺序，首条前向标量完全相同但初始裁剪后分组梯度略有差异，不宣称反向逐位一致。

3075/warmup93与4305/warmup130，实际分组LR由已记录P20 LR和组基础LR重建并验证step−1线性因子。最早存档loss在update20已分岔3.89994/4.22554，裁剪前梯度9.038/23.337；update21 P20 RMS .138710/.097986、偏移.244323/.227782，而摘要/融合熵差极小。第1轮5/11/17层权重均值五轮.181/.337/.482、七轮.277/.375/.348。七轮第5轮仍约29.49%基础LR，非原五轮续训；同期loss窗口均值.583579高于原五轮.561589。e5 Overall56.4307对旧58.6835下降2.2528pp，已有配对CI[-3.1200,-1.3895]（引用旧统计，不重跑）。

七轮e5/6/7：Overall56.4307/55.7437/55.8396；YesNo89.6000/89.3120/89.9520；FF23.3567/22.2719/21.8251；what19.1130/17.6609/17.3862；where56.7237/57.2127/55.5012。同期loss均值.583579/.539088/.501028；融合熵.640369/.616685/.629579；偏移/P20 .358275/.367037/.355035；LN后约1。裁剪前梯度中位数.538864/.624449/.711785，超过1的记录1/30、2/31、3/31，不是全步裁剪率。各组裁剪后梯度另图展示，不与Trainer裁剪前混用。

结论：前五轮已走不同优化轨迹，后两轮loss继续下降但开放回答继续恶化；无持续地图集中、比例爆炸或长期裁剪限制的直接证据。调度/早期前缀与层门控分岔是线索，不能单独证实过拟合、定位失败或具体LR根因。所有激活来自变化minibatch，缺固定样本向量，不补模型。产物 `pathvqa/outputs/v10_5ep_7ep_cpu_comparison_20261009/`：10张叠加图、index.html、report.md、comparison.json（逐轮分位数/全部梯度组/协议/来源SHA256/缺失项）。相关CPU脚本diagnostics/compare_v10_archived_curves.py；本次结束，不启动训练或增加计划；账本按规则随下次相关代码提交。

## 2026-10-09 V10条件头修正版7轮完成：负结果

实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_7ep_seed44`，PathVQA model seed44/data seed42，从零7轮，保存及完整Validation评估5/6/7（6259题/832图），固定epoch7主结果。唯一配置改动为预算5→7及随总预算重排的3%warmup/linear；不等于原5轮checkpoint续训。结构、初始化、分组基础LR、batch2/累积16、归一化False保持，30个共有初值核验一致。

| checkpoint | Overall | Yes/No | Free-form | what | where |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch5 | 56.4307 | 89.6000 | 23.3567 | 19.1130 | 56.7237 |
| epoch6 | 55.7437 | 89.3120 | 22.2719 | 17.6609 | 57.2127 |
| epoch7（主） | 55.8396 | 89.9520 | 21.8251 | 17.3862 | 55.5012 |

对原修正版5轮58.6835：本次epoch5 Overall -2.252756pp，图像簇95%配对CI[-3.119999,-1.389499]；epoch6 -2.939767，CI[-3.873298,-2.017625]；epoch7 -2.843905，CI[-3.790938,-1.914693]。epoch7 Yes/No -0.2880，CI[-1.277142,0.705168]；Free-form -5.392470，CI[-6.990996,-3.816776]；what -4.709576，CI[-6.383886,-3.046664]；where -11.491443，CI[-16.381418,-6.811418]。对原V1五轮seed44 Overall -3.498961，CI[-4.414125,-2.582697]。配对10000次/seed42；分项探索性。epoch7独立95%CI[54.3426,57.3278]，2000次/seed42/832图；其他所有分轮、分项、配对精确数值及运行元数据保留在 `pathvqa/outputs/cpu_curve_audit_20261009/v10_7ep_user_result.json`（用户附件首条完整JSON）。

参数1691043，训练15073.5807秒，平均train_loss0.6558331658477429，峰值25000120320bytes；commit b2724f30e18f8b559fa97dd6550677953a87226c；Torch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0。输出根 `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_7ep_seed44_20261009_151030_637257`；checkpoint `checkpoints/epoch_N`，预测/summary `eval_validation/epoch_N/pathvqa_predictions.json`及`pathvqa_summary.json`。

结论：七轮配置失败，主要损失来自开放回答；在epoch5已劣于原5轮计划，不能只归因额外两轮。5→7轮开放回答继续下降，但缺逐步训练loss/梯度/地图诊断，尚不能确定过拟合、优化路径/尺度或地图集中等机制；不能据此宣称Meta-Net LR3e-4本身过高或共同Value是根因。保留原五轮修正版及V1，停止加轮数；余下一试待用户选择。助手仅CPU读取/账本记录，无GPU操作，无仅账本commit/push。

## 2026-10-09 V10条件头修正版：既有epoch3/4完整Validation补评估完成

实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`；PathVQA model seed44/data seed42。用户回传既有epoch3/4 checkpoint完整Validation结果，6259题/832图；仅改变被评估checkpoint轮数，没有新训练/结构/监督改动，epoch5仍为预定主结果。输出根 `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548/epoch3_4_validation_20261009_143119`；各轮 `epoch_3` / `epoch_4` 下保存pathvqa_predictions.json及pathvqa_summary.json。epoch5复用原eval_validation/epoch_5产物，不重新评分。

| Epoch | Overall | Yes/No | Free-form | how | other | what | when | where | why |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 57.1337 | 89.0880 | 25.2712 | 9.3023 | 7.1429 | 19.5055 | 0 | 68.7042 | 4.7619 |
| 4 | 58.0284 | 89.9200 | 26.2285 | 8.5271 | 14.2857 | 21.0361 | 0 | 66.5037 | 4.7619 |
| 5（既有主结果） | 58.6835 | 90.2400 | 27.2176 | 10.0775 | 14.2857 | 22.0958 | 0 | 66.9927 | 4.7619 |

epoch4独立图像簇95%CI[56.60,59.46]（用户显示精度），832簇；未回传epoch3 CI或轮间配对统计，不补造。epoch4 TTFT mean/p50/p95/min/max=.051842/.051475/.054881/.036814/.306499秒；TPOT=.019864秒/token，50.342token/s；request mean/p50/p95/min/max=.103692/.0913/.154115/.061564/.455957秒，6259请求。

诊断：Overall逐轮+0.8947/+0.6551pp，Free-form+0.9573/+0.9891pp，what+1.5306/+1.0597pp；where68.7042→66.5037→66.9927，并非各子组同步改善。结合既有训练loss下降，支持该运行到epoch5仍有验证收益，尚未观察到Overall在3–5轮转为过拟合；不证明epoch6+继续提升、不证明加大学习率有效，不把轮间点差当显著差异。单seed且检查了Validation，后续延长属于新增调参实验，应另行授权与记录。助手只本机记录/读取，无GPU操作；不创建仅账本提交或推送。

# MMRL Experiment Ledger

## 2026-10-09 条件头修正版七轮：代码准备，尚未运行

实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_7ep_seed44`，PathVQA44/42，从零训练；唯一改变总预算5→7，3%warmup/线性衰减随七轮终点重算。结构/共有初值/分组基础LR（Meta3e-4、LN1e-4、P20.3、S8 3e-5、其余1e-4）、batch2/累积16、AdamW/clip1/监督/mask和归一化False均沿用修正版。参数1,691,043。保存并完整Validation评估epoch5/6/7，epoch7预定主结果，不挑最高轮次；分别对既有修正版五轮58.6835及V1五轮59.3386配对图像簇10000/seed42，五个既定分项，子组与5/6轮趋势探索性。独立预算/保存/回调/报告配置，不只是改实验名。

输出 `pathvqa/outputs/v10_head_fixed/<7ep_experiment>_<timestamp>/`，实际目录/提交/训练成本/分数待用户执行回传。保留原训练诊断，CPU绘图从真实state.max_steps及报告warmup比例取坐标，不写死3075/93，末轮loss窗口亦动态。Windows本机语法/Bash及复用既有五轮日志绘图检查通过；没有七轮张量运行或新成绩。启动 `bash pathvqa/run_v10_head_fixed_7ep_seed44.sh`，所有GPU用户执行；失败即停，不重试/关机/Test/其他seed，不自动安排最后候选。结果和失败回传后追加两账本及计划，不覆盖旧五轮产物。

## 2026-10-09 条件头修正版训练曲线CPU只读分析完成

绑定 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548`，目录位于服务器pathvqa/outputs/v10_head_fixed；PathVQA44/42，固定epoch5 Validation已有Overall58.6835/Yes-No90.2400/Free-form27.2176，本次未重新评分或推理。SSH只读复制4份日志到本机 `pathvqa/outputs/cpu_curve_audit_20261009/`，生成四组PNG、index.html、report.md、curve_analysis.json与重建LR CSV；源码 `diagnostics/plot_v10_training_logs.py` 不导入torch/CUDA。执行commit4838875e385650d121319599fe776a1f5215733e，实际参数1,691,043、3075步/五轮、训练10773.9051秒、日志峰值23.2832GiB；Torch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0，归一化False、Trainer16/Accelerate1，共有30张量初值核验已记录通过。未执行任何GPU/服务器写入。

- loss/LR：93步warmup；实际optimizer日志与报告基础LR一致，P20 .3/S8 3e-5/Meta-Net3e-4/其余1e-4。Trainer记录的是第一组P20；共同factor重建分组LR，与线性scheduler step−1位置吻合（误差1.11e-16），不是直接逐组逐步LR观测。按日志epoch分组的20步窗口均值0.9361/.7110/.6591/.6076/.5616；末轮前半.5752、后半.5480，仍缓慢下降，不能保证多训Validation收益。
- 梯度：裁剪前总范数与裁剪后组范数分图；warmup后149个采样点1个>1（0.67%），中位数.4359、P90 .6188、max1.6065，未见长期clip限制。每20步采样非完整步史；callback rawstep k对应更新k+1，不混为同一步。Meta-Net组范数较大不等于性能或AdamW更新贡献最大。
- 尺度：LN前RMS中位数.4699、LN后.9996；偏移/P20初始化9.435迅速降低，末次.3232、中位数.3990。偏移RMS .2049→1.4463，P20 .0217→4.4749；无后期持续尺度爆炸证据。
- 地图：有效融合merger网格归一化熵.99998→.6410，中位数.7677；层17均重.5207，末次层5/11/17=.2703/.2297/.5000。形成非均匀选择，不证明定位有效或层17贡献最大。摘要RMS .4180→.4857，但变化mini-batch混杂，缺少向量/固定样本轨迹，不能分析真实摘要方向变化。逐层合并地图熵、逐步组LR和全步裁剪亦未保存，明确缺失，不补跑。

本次仅轨迹描述，无新训练/消融/seed/Test；不直接归因为优化失稳，不自动安排延长训练。来源SHA256及完整数值分位数保存在本机curve_analysis.json；历史成绩不改。

## 2026-10-09 V10条件头修正版完成：PathVQA seed44固定epoch5 Validation

实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`；用户回传完整summary、两组配对与train_report。本条覆盖此前同名“仅准备/尚未运行”状态，不删除历史。PathVQA model seed44/data seed42，19654 Train样本，从零5轮，保存3/4/5，固定epoch5完整Validation6259题/832图，不跑Test/其他seed。受控改动：原V10融合摘要增加LN2560、Meta-Net输出恢复默认Linear权重与bias初始化、Meta-Net LR1e-4→3e-4，新增LN LR1e-4；地图/Visual18/P20及其他协议保持，三项组合不能分别归因。

成绩：Overall **58.68349576609682**（显示58.6835），Yes/No90.2400，Free-form27.2176；how10.0775/other14.2857/what22.0958/when0/where66.9927/why4.7619/yes-no90.24。Free-form题型macro19.7023。独立图像簇95%CI[57.2044,60.1183]，2000次/seed42。

| 对照 | 分组 | delta(pp) | 95%图像簇配对CI | 修正版独占/对照独占 | McNemar exact p | 探索性 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| v10 | overall | 0.9745965809 | [0.1117091179, 1.8222379228] | 360/299 | 0.019356622451895497 | False |
| v10 | yes_no | 0.6080000000 | [-0.3599535370, 1.6394102372] | 133/114 | 0.25203054645422135 | True |
| v10 | free_form | 1.3401403957 | [-0.0325849666, 2.7103467083] | 227/185 | 0.043260411200230275 | True |
| v10 | what | 1.2951334380 | [-0.0404537592, 2.6650884289] | 166/133 | 0.06404566585485909 | True |
| v10 | where | 1.7114914425 | [-3.1784841076, 6.6176470588] | 56/49 | 0.5583942911655908 | True |
| v1 | overall | -0.6550567183 | [-1.5107916235, 0.1788595540] | 286/327 | 0.1061075986504401 | False |
| v1 | yes_no | -0.6400000000 | [-1.6594797352, 0.3852482284] | 107/127 | 0.21412325762860127 | True |
| v1 | free_form | -0.6700701978 | [-2.0395161150, 0.6974725045] | 179/200 | 0.30425909881582563 | True |
| v1 | what | 0.0392464678 | [-1.3991943652, 1.4853564260] | 149/148 | 1.0 | True |
| v1 | where | -5.3789731051 | [-9.5354523227, -1.2254901961] | 26/48 | 0.014079981429530057 | True |

配对均为10000次、seed42、image_id簇；Overall6259题/832簇，Yes/No3125/810，Free-form3134/821，what2548/817，where409/408。原V10 seed44 Overall57.70889918517335、原V1 seed44 Overall59.33855248442243；相对V10净增61题，相对V1净少41题。

运行：commit `4838875e385650d121319599fe776a1f5215733e`；torch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0；实际参数1,691,043，训练10773.9051秒，train_loss0.6936398631770436，峰值GPU 25000120320bytes（23.283176GiB）。30个共有初值核验一致；输出weight实际std0.0456634499，bias RMS0.0456702374，summary LN默认weight1/bias0/eps1e-5。

协议：bf16、max_length2048、workers2，AdamW betas(.9,.999)/eps1e-8/WD0，batch2/accum16、warmup3%/linear、clip1；model_accepts_loss_kwargs=False、Accelerate内部累积1，等权microbatch均值归一化。P20 .3/S8 3e-5/Av10 1e-4/question_context 1e-4/maps 1e-4/layer_condition 1e-4/Meta-Net 3e-4/LN 1e-4。

输出根：`/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548`；checkpoint `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548/checkpoints/epoch_5`；`eval_validation/epoch_5/pathvqa_predictions.json`及`pathvqa_summary.json`。TTFT mean0.050917s，TPOT0.019453s/token（51.406token/s），request mean0.101243s；跨运行硬件/环境未经统一核验，不据此宣称速度变化。

解释：该配置相对原V10 Overall配对CI低端为+0.1117，支持同seed整体改善；相对V1 Overall点估计仍低0.6551pp、CI跨零，不等于非劣或性能相同。相对V1的where下降5.3790pp、探索性CI低于零。尚不能确认学习率是主要原因、地图瓶颈已解决或跨seed稳定。下一步优先读取既有trainer/trainer_state.json和v10_diagnostics.jsonl绘制loss/LR/裁剪前总梯度与裁剪后分组梯度/LN与偏移尺度/地图有效熵；本次未回传轨迹，未绘制实际曲线，不自动安排新训练。

<details>
<summary>完整用户回传原文（含所有精确统计和运行元数据）</summary>

~~~~text
========== PathVQA Evaluation ==========
Overall Accuracy: 58.68
Yes/No Accuracy: 90.24
Free-form Accuracy: 27.22
Per Question Type: {'how': 10.0775, 'other': 14.2857, 'what': 22.0958, 'when': 0.0, 'where': 66.9927, 'why': 4.7619, 'yes/no': 90.24}
Image-clustered 95% CI: [57.20, 60.12] clusters=832
TTFT: {'count': 6259, 'mean': 0.050917, 'p50': 0.050641, 'p95': 0.053224, 'min': 0.036142, 'max': 0.33747}
TPOT: 0.019453 s/token, 51.406 token/s
Request Latency: {'count': 6259, 'mean': 0.101243, 'p50': 0.08928, 'p95': 0.149458, 'min': 0.060399, 'max': 0.430377}
Predictions: /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548/eval_validation/epoch_5/pathvqa_predictions.json
Summary: /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548/eval_validation/epoch_5/pathvqa_summary.json
[PATHVQA_PAIRED_COMPARISON] group=overall n=6259 clusters=832 baseline=57.7089 variant=58.6835 delta=+0.9746 variant_only=360 baseline_only=299 ci={'confidence_level': 0.95, 'lower': 0.11170911785567676, 'upper': 1.8222379228497967, 'iterations': 10000, 'seed': 42, 'image_clusters': 832}
[PATHVQA_PAIRED_COMPARISON] group=yes_no n=3125 clusters=810 baseline=89.6320 variant=90.2400 delta=+0.6080 variant_only=133 baseline_only=114 ci={'confidence_level': 0.95, 'lower': -0.359953537031513, 'upper': 1.639410237192952, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
[PATHVQA_PAIRED_COMPARISON] group=free_form n=3134 clusters=821 baseline=25.8775 variant=27.2176 delta=+1.3401 variant_only=227 baseline_only=185 ci={'confidence_level': 0.95, 'lower': -0.03258496664800933, 'upper': 2.710346708334561, 'iterations': 10000, 'seed': 42, 'image_clusters': 821}
[PATHVQA_PAIRED_COMPARISON] group=what n=2548 clusters=817 baseline=20.8006 variant=22.0958 delta=+1.2951 variant_only=166 baseline_only=133 ci={'confidence_level': 0.95, 'lower': -0.040453759151467414, 'upper': 2.6650884288694137, 'iterations': 10000, 'seed': 42, 'image_clusters': 817}
[PATHVQA_PAIRED_COMPARISON] group=where n=409 clusters=408 baseline=65.2812 variant=66.9927 delta=+1.7115 variant_only=56 baseline_only=49 ci={'confidence_level': 0.95, 'lower': -3.1784841075794623, 'upper': 6.617647058823529, 'iterations': 10000, 'seed': 42, 'image_clusters': 408}
[PATHVQA_PAIRED_COMPARISON] group=overall n=6259 clusters=832 baseline=59.3386 variant=58.6835 delta=-0.6551 variant_only=286 baseline_only=327 ci={'confidence_level': 0.95, 'lower': -1.5107916234561787, 'upper': 0.17885955401498577, 'iterations': 10000, 'seed': 42, 'image_clusters': 832}
[PATHVQA_PAIRED_COMPARISON] group=yes_no n=3125 clusters=810 baseline=90.8800 variant=90.2400 delta=-0.6400 variant_only=107 baseline_only=127 ci={'confidence_level': 0.95, 'lower': -1.6594797352123056, 'upper': 0.38524822841452755, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
[PATHVQA_PAIRED_COMPARISON] group=free_form n=3134 clusters=821 baseline=27.8877 variant=27.2176 delta=-0.6701 variant_only=179 baseline_only=200 ci={'confidence_level': 0.95, 'lower': -2.039516115001131, 'upper': 0.6974725044899934, 'iterations': 10000, 'seed': 42, 'image_clusters': 821}
[PATHVQA_PAIRED_COMPARISON] group=what n=2548 clusters=817 baseline=22.0565 variant=22.0958 delta=+0.0392 variant_only=149 baseline_only=148 ci={'confidence_level': 0.95, 'lower': -1.3991943652084513, 'upper': 1.4853564259907093, 'iterations': 10000, 'seed': 42, 'image_clusters': 817}
[PATHVQA_PAIRED_COMPARISON] group=where n=409 clusters=408 baseline=72.3716 variant=66.9927 delta=-5.3790 variant_only=26 baseline_only=48 ci={'confidence_level': 0.95, 'lower': -9.535452322738386, 'upper': -1.2254901960784315, 'iterations': 10000, 'seed': 42, 'image_clusters': 408}
[V10_HEAD_FIXED_DONE] {"summary": {"count": 6259, "overall_accuracy": 58.6835, "yes_no_accuracy": 90.24, "free_form_accuracy": 27.2176, "per_answer_type_accuracy": {"free-form": 27.2176, "yes/no": 90.24}, "per_question_type_accuracy": {"how": 10.0775, "other": 14.2857, "what": 22.0958, "when": 0.0, "where": 66.9927, "why": 4.7619, "yes/no": 90.24}, "free_form_question_type_macro_accuracy": 19.7023, "clustered_bootstrap": {"point_estimate": 58.6835, "confidence_level": 0.95, "lower": 57.2044, "upper": 60.1183, "iterations": 2000, "seed": 42, "image_clusters": 832}, "backend": "v10-head-fixed", "base_model": "/root/autodl-tmp/model", "checkpoint": "/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548/checkpoints/epoch_5", "v0_intervention": "normal", "v0_question_mask_policy": "independent_raw_question_ids_no_prefill_write_mapping", "question_source_policy": "independent_raw_question_ids_no_prefill_write_mapping", "dynamic_prompt_component_checkpoint": null, "dynamic_prompt_components": [], "data_root": "/root/autodl-tmp/dataset/pathVQA", "split": "validation", "max_new_tokens": 32, "temperature": 0.0, "answer_mode": "raw", "instruction": "short-answer", "partial_evaluation": false, "timing": {"methodology": {"warmup_runs_excluded": 3, "ttft": "generate start to first generated-token logits ready", "tpot": "token-count-weighted interval between later token logits", "request": "model interface call including preprocessing and decoding", "timing_methods": {"cuda-events-logits-ready-v2": 6259}}, "successful_requests": 6259, "model_timed_requests": 6259, "generated_tokens": 17923, "subsequent_tokens": 11664, "ttft_seconds": {"count": 6259, "mean": 0.050917, "p50": 0.050641, "p95": 0.053224, "min": 0.036142, "max": 0.33747}, "tpot_per_request_seconds": {"count": 6259, "mean": 0.019474, "p50": 0.01943, "p95": 0.019749, "min": 0.018821, "max": 0.035289}, "tpot_weighted_seconds": 0.019453, "decode_tokens_per_second": 51.406, "generation_seconds": {"count": 6259, "mean": 0.087348, "p50": 0.071151, "p95": 0.133919, "min": 0.055173, "max": 0.418857}, "request_seconds": {"count": 6259, "mean": 0.101243, "p50": 0.08928, "p95": 0.149458, "min": 0.060399, "max": 0.430377}, "model_generated_tokens_per_second": 32.783, "end_to_end_generated_tokens_per_second": 28.284}}, "comparisons": {"v10": {"experiment": "pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44", "baseline": "pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44", "bootstrap": {"unit": "image_id", "iterations": 10000, "seed": 42}, "validation_summary": {"count": 6259, "overall_accuracy": 58.6835, "yes_no_accuracy": 90.24, "free_form_accuracy": 27.2176, "per_answer_type_accuracy": {"free-form": 27.2176, "yes/no": 90.24}, "per_question_type_accuracy": {"how": 10.0775, "other": 14.2857, "what": 22.0958, "when": 0.0, "where": 66.9927, "why": 4.7619, "yes/no": 90.24}, "free_form_question_type_macro_accuracy": 19.7023, "clustered_bootstrap": {"point_estimate": 58.6835, "confidence_level": 0.95, "lower": 57.2044, "upper": 60.1183, "iterations": 2000, "seed": 42, "image_clusters": 832}, "backend": "v10-head-fixed", "base_model": "/root/autodl-tmp/model", "checkpoint": "/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548/checkpoints/epoch_5", "v0_intervention": "normal", "v0_question_mask_policy": "independent_raw_question_ids_no_prefill_write_mapping", "question_source_policy": "independent_raw_question_ids_no_prefill_write_mapping", "dynamic_prompt_component_checkpoint": null, "dynamic_prompt_components": [], "data_root": "/root/autodl-tmp/dataset/pathVQA", "split": "validation", "max_new_tokens": 32, "temperature": 0.0, "answer_mode": "raw", "instruction": "short-answer", "partial_evaluation": false, "timing": {"methodology": {"warmup_runs_excluded": 3, "ttft": "generate start to first generated-token logits ready", "tpot": "token-count-weighted interval between later token logits", "request": "model interface call including preprocessing and decoding", "timing_methods": {"cuda-events-logits-ready-v2": 6259}}, "successful_requests": 6259, "model_timed_requests": 6259, "generated_tokens": 17923, "subsequent_tokens": 11664, "ttft_seconds": {"count": 6259, "mean": 0.050917, "p50": 0.050641, "p95": 0.053224, "min": 0.036142, "max": 0.33747}, "tpot_per_request_seconds": {"count": 6259, "mean": 0.019474, "p50": 0.01943, "p95": 0.019749, "min": 0.018821, "max": 0.035289}, "tpot_weighted_seconds": 0.019453, "decode_tokens_per_second": 51.406, "generation_seconds": {"count": 6259, "mean": 0.087348, "p50": 0.071151, "p95": 0.133919, "min": 0.055173, "max": 0.418857}, "request_seconds": {"count": 6259, "mean": 0.101243, "p50": 0.08928, "p95": 0.149458, "min": 0.060399, "max": 0.430377}, "model_generated_tokens_per_second": 32.783, "end_to_end_generated_tokens_per_second": 28.284}}, "groups": {"overall": {"count": 6259, "baseline_accuracy": 57.70889918517335, "variant_accuracy": 58.68349576609682, "delta": 0.9745965809234676, "variant_only_correct": 360, "baseline_only_correct": 299, "mcnemar_exact_p": 0.019356622451895497, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": 0.11170911785567676, "upper": 1.8222379228497967, "iterations": 10000, "seed": 42, "image_clusters": 832}, "question_count": 6259, "image_clusters": 832, "exploratory": false}, "yes_no": {"count": 3125, "baseline_accuracy": 89.632, "variant_accuracy": 90.24, "delta": 0.6079999999999899, "variant_only_correct": 133, "baseline_only_correct": 114, "mcnemar_exact_p": 0.25203054645422135, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -0.359953537031513, "upper": 1.639410237192952, "iterations": 10000, "seed": 42, "image_clusters": 810}, "question_count": 3125, "image_clusters": 810, "exploratory": true}, "free_form": {"count": 3134, "baseline_accuracy": 25.87747287811104, "variant_accuracy": 27.217613273771537, "delta": 1.340140395660498, "variant_only_correct": 227, "baseline_only_correct": 185, "mcnemar_exact_p": 0.043260411200230275, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -0.03258496664800933, "upper": 2.710346708334561, "iterations": 10000, "seed": 42, "image_clusters": 821}, "question_count": 3134, "image_clusters": 821, "exploratory": true}, "what": {"count": 2548, "baseline_accuracy": 20.800627943485086, "variant_accuracy": 22.09576138147567, "delta": 1.2951334379905823, "variant_only_correct": 166, "baseline_only_correct": 133, "mcnemar_exact_p": 0.06404566585485909, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -0.040453759151467414, "upper": 2.6650884288694137, "iterations": 10000, "seed": 42, "image_clusters": 817}, "question_count": 2548, "image_clusters": 817, "exploratory": true}, "where": {"count": 409, "baseline_accuracy": 65.28117359413203, "variant_accuracy": 66.99266503667482, "delta": 1.711491442542794, "variant_only_correct": 56, "baseline_only_correct": 49, "mcnemar_exact_p": 0.5583942911655908, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -3.1784841075794623, "upper": 6.617647058823529, "iterations": 10000, "seed": 42, "image_clusters": 408}, "question_count": 409, "image_clusters": 408, "exploratory": true}}}, "v1": {"experiment": "pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44", "baseline": "pathvqa_v1_norm_fixed_5ep_seed44", "bootstrap": {"unit": "image_id", "iterations": 10000, "seed": 42}, "validation_summary": {"count": 6259, "overall_accuracy": 58.6835, "yes_no_accuracy": 90.24, "free_form_accuracy": 27.2176, "per_answer_type_accuracy": {"free-form": 27.2176, "yes/no": 90.24}, "per_question_type_accuracy": {"how": 10.0775, "other": 14.2857, "what": 22.0958, "when": 0.0, "where": 66.9927, "why": 4.7619, "yes/no": 90.24}, "free_form_question_type_macro_accuracy": 19.7023, "clustered_bootstrap": {"point_estimate": 58.6835, "confidence_level": 0.95, "lower": 57.2044, "upper": 60.1183, "iterations": 2000, "seed": 42, "image_clusters": 832}, "backend": "v10-head-fixed", "base_model": "/root/autodl-tmp/model", "checkpoint": "/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548/checkpoints/epoch_5", "v0_intervention": "normal", "v0_question_mask_policy": "independent_raw_question_ids_no_prefill_write_mapping", "question_source_policy": "independent_raw_question_ids_no_prefill_write_mapping", "dynamic_prompt_component_checkpoint": null, "dynamic_prompt_components": [], "data_root": "/root/autodl-tmp/dataset/pathVQA", "split": "validation", "max_new_tokens": 32, "temperature": 0.0, "answer_mode": "raw", "instruction": "short-answer", "partial_evaluation": false, "timing": {"methodology": {"warmup_runs_excluded": 3, "ttft": "generate start to first generated-token logits ready", "tpot": "token-count-weighted interval between later token logits", "request": "model interface call including preprocessing and decoding", "timing_methods": {"cuda-events-logits-ready-v2": 6259}}, "successful_requests": 6259, "model_timed_requests": 6259, "generated_tokens": 17923, "subsequent_tokens": 11664, "ttft_seconds": {"count": 6259, "mean": 0.050917, "p50": 0.050641, "p95": 0.053224, "min": 0.036142, "max": 0.33747}, "tpot_per_request_seconds": {"count": 6259, "mean": 0.019474, "p50": 0.01943, "p95": 0.019749, "min": 0.018821, "max": 0.035289}, "tpot_weighted_seconds": 0.019453, "decode_tokens_per_second": 51.406, "generation_seconds": {"count": 6259, "mean": 0.087348, "p50": 0.071151, "p95": 0.133919, "min": 0.055173, "max": 0.418857}, "request_seconds": {"count": 6259, "mean": 0.101243, "p50": 0.08928, "p95": 0.149458, "min": 0.060399, "max": 0.430377}, "model_generated_tokens_per_second": 32.783, "end_to_end_generated_tokens_per_second": 28.284}}, "groups": {"overall": {"count": 6259, "baseline_accuracy": 59.33855248442243, "variant_accuracy": 58.68349576609682, "delta": -0.6550567183256106, "variant_only_correct": 286, "baseline_only_correct": 327, "mcnemar_exact_p": 0.1061075986504401, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -1.5107916234561787, "upper": 0.17885955401498577, "iterations": 10000, "seed": 42, "image_clusters": 832}, "question_count": 6259, "image_clusters": 832, "exploratory": false}, "yes_no": {"count": 3125, "baseline_accuracy": 90.88, "variant_accuracy": 90.24, "delta": -0.6400000000000006, "variant_only_correct": 107, "baseline_only_correct": 127, "mcnemar_exact_p": 0.21412325762860127, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -1.6594797352123056, "upper": 0.38524822841452755, "iterations": 10000, "seed": 42, "image_clusters": 810}, "question_count": 3125, "image_clusters": 810, "exploratory": true}, "free_form": {"count": 3134, "baseline_accuracy": 27.887683471601786, "variant_accuracy": 27.217613273771537, "delta": -0.670070197830249, "variant_only_correct": 179, "baseline_only_correct": 200, "mcnemar_exact_p": 0.30425909881582563, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -2.039516115001131, "upper": 0.6974725044899934, "iterations": 10000, "seed": 42, "image_clusters": 821}, "question_count": 3134, "image_clusters": 821, "exploratory": true}, "what": {"count": 2548, "baseline_accuracy": 22.05651491365777, "variant_accuracy": 22.09576138147567, "delta": 0.03924646781789676, "variant_only_correct": 149, "baseline_only_correct": 148, "mcnemar_exact_p": 1.0, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -1.3991943652084513, "upper": 1.4853564259907093, "iterations": 10000, "seed": 42, "image_clusters": 817}, "question_count": 2548, "image_clusters": 817, "exploratory": true}, "where": {"count": 409, "baseline_accuracy": 72.37163814180929, "variant_accuracy": 66.99266503667482, "delta": -5.378973105134463, "variant_only_correct": 26, "baseline_only_correct": 48, "mcnemar_exact_p": 0.014079981429530057, "clustered_paired_delta_ci": {"confidence_level": 0.95, "lower": -9.535452322738386, "upper": -1.2254901960784315, "iterations": 10000, "seed": 42, "image_clusters": 408}, "question_count": 409, "image_clusters": 408, "exploratory": true}}}}, "train_report": {"experiment": "pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44", "method": "visual_selection_v10_condition_head_fixed", "dataset": "PathVQA", "model_seed": 44, "data_seed": 42, "epochs": 5, "saved_epochs": [3, 4, 5], "train_split": "train", "languages": "all", "train_samples": 19654, "train_manifest": null, "base_model": "/root/autodl-tmp/model", "max_length": 2048, "workers": 2, "bf16": true, "trainable_parameters": {"p20": 51200, "visual_s8": 8192, "visual_av10": 10240, "question_context": 345088, "maps": 448896, "layer_condition": 387, "meta_net": 821920, "summary_norm": 5120}, "total_trainable_parameters": 1691043, "git_commit": "4838875e385650d121319599fe776a1f5215733e", "runtime_versions": {"torch": "2.8.0+cu128", "transformers": "5.0.0", "accelerate": "1.12.0"}, "train_metrics": {"train_runtime": 10773.9051, "train_samples_per_second": 9.121, "train_steps_per_second": 0.285, "total_flos": 0.0, "train_loss": 0.6936398631770436, "epoch": 5.0}, "peak_gpu_memory_bytes": 25000120320, "initialization": {"reference": "same_seed_original_V10", "shared_tensor_count": 30, "all_shared_initial_values_equal": true, "visual18_order": "S8_then_Av10", "visual18_init": "Normal(0,0.02)", "head_rng": "private_CPU_seed_equals_model_seed_global_state_restored", "meta_first_init": "nn.Linear_default", "meta_output_init": "nn.Linear_default", "meta_output_bias_zero": false, "meta_output_actual_std": 0.04566344991326332, "meta_output_bias_rms": 0.04567023739218712, "summary_norm_init": "LayerNorm_default_weight1_bias0_eps1e-5"}, "trainer_model_accepts_loss_kwargs": false, "accelerator_gradient_accumulation_steps": 1, "loss_accumulation_protocol": "equal_microbatch_mean_trainer_normalized", "optimizer": {"type": "AdamW", "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0, "per_device_batch_size": 2, "gradient_accumulation_steps": 16, "warmup_ratio": 0.03, "scheduler": "linear", "max_grad_norm": 1.0, "group_learning_rates": {"p20": 0.3, "visual_s8": 3e-05, "visual_av10": 0.0001, "question_context": 0.0001, "maps": 0.0001, "layer_condition": 0.0001, "meta_net": 0.0003, "summary_norm": 0.0001}}}, "controlled_change": "LN + default Meta-Net initialization + Meta-Net LR3e-4 combined"}
~~~~
</details>


## 2026-10-09 V10条件头修正版准备（未运行，无新成绩）

`pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`：PathVQA model44/data42，从零5 epochs，固定epoch5完整Validation；仅组合摘要LN2560、Meta-Net两层默认Linear权重/bias初始化、Meta-Net LR3e-4（LN LR1e-4），原V10其余结构/共有初值/各组LR/监督/累积归一化及训练预算不变。预期1,691,043参数，启动模型/优化器按组核算；没有把CPU算术当作真实GPU实测。独立模型/config/weights/backend/入口，原模型保持原行为。相对既有V10 57.7089、V1 59.3386各自图像簇配对10000次/seed42；子组探索性，不能把组合收益归因单项。输出 `pathvqa/outputs/v10_head_fixed/<experiment>_<timestamp>/`，确切目录/执行commit/成绩/成本/预检实测待用户回传。保留地图/层权重/偏移与梯度诊断，补LN前后摘要RMS；首批与Trainer grad_norm裁剪前、组梯度回调裁剪后。CPU语法及参数算术检查；共有参数逐张量核对、真实前向梯度与cache检查由用户启动既有预检。无Test/其他seed/重试/关机，剩余候选未实施。

## 2026-10-08 PathVQA V10三seed代码准备（不是已完成实验）

实验 `pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44`、seed45、seed46，尚无分数。保持已跑通V10结构/初始化/学习率，迁移到PathVQA；各自从零五轮，data42、batch2/累积16、归一化False，保存3/4/5，固定epoch5完整Validation，无Test/额外seed/调参。参数1,685,923。精确绑定同seed五轮V1既有产物（44目录20260926_2，45/46目录20260926），复用图像簇配对10000/seed42及全部题型；三项成功后mean±样本std(ddof=1)。输出 `pathvqa/outputs/v10/suite_<timestamp>/`。失败即停且保存错误退出码/未运行队列；本次专用入口用户启动后，无论成功失败均先落盘报告，再独立计时600秒关机，关机失败单列。实现和CPU检查不代表GPU预检/训练/评估已完成。结果回传后补精确分项/成本/提交与真实目录。

## 2026-10-01 SSH只读补齐原五轮V1产物身份

连接`connect.bjb1.seetacloud.com:50241`，只读JSON、日志、文件列表与torch.save ZIP CRC，无torch导入、模型前向、GPU调用或服务器文件写入。修复后的本机校验函数在内存中对真实文件执行，三个PathVQA身份校验及两套RSVQA完整既有评分校验均通过；这不是新训练或Test评估。

以下路径位于`pathvqa/outputs/visual_selection_prefix/`，均固定`checkpoints/epoch_5`，原V1 method、1864963参数、model seed如下/data42、5epochs、保存3/4/5、batch2/累积16、loss kwargs=False、Accelerate累积1。三个历史运行均明确记录torch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0；原日志确认P20 .3、S8 3e-5、Av10 1e-4，其余条件组1e-4。

| seed | 精确运行目录 | 实际训练提交 | 训练秒 | 报告峰值显存bytes | 已有Validation Overall |
|---|---|---|---:|---:|---:|
|44|pathvqa_v1_norm_fixed_5ep_seed44_20260926_2|2957ce297637d65a8dcdb5a075c1528072497590|10863.0715|25011373568|59.3386|
|45|pathvqa_v1_norm_fixed_5ep_seed45_20260926|f3b9c3922ec44f3388a43b6fc9cf50d28bf30672|10667.1862|25011373568|58.5077|
|46|pathvqa_v1_norm_fixed_5ep_seed46_20260926|548335ffa98594b478117139c80a5ff128cff8ce|10857.1599|25011373568|58.7314|

明确更正此前seed45/46路径/提交/版本“未知”状态：现由实际产物证实，历史分数不变。seed44前两个无后缀/`_1`目录无train_report，绑定完成的`_2`，不按最新选择。epoch5权重SHA256：44=`18ee0c0f2f3b41f107752b2d160c9c5154bc70b3d8366e5e04a7ba9df8e64aa2`，45=`ec5de2aec77e0783c340207ac51f4ef85f1af7ae5dc5c506170420583efd2796`，46=`ee2915d28a6b8b933cb51fef7682c4d467fe6b3c2651466d4db761b9704e11da`。入口已改为这三个精确路径，仍执行身份与归一化核验。

RSVQA V1 `rsvqa_v1_norm_fixed_5ep_b4a8_seed44_20260929`报告确认五轮、batch4/累积8、1864963参数，训练提交`1ae3d1362dddd9d9ec02074eb9bb45c24325e03e`、上述相同版本、训练10982.6985秒、报告峰值12892518912bytes。既有Test10004题/100图、OA85.0360/AA86.0812评分复核通过。RSVQA LoRA `rsvqa_lr_lora_full_model_attention_r8_b4a8_seed44_20260929`报告确认三轮/r8 alpha16/7077888参数/batch4累积8/归一化False、训练12284.0502秒；既有Test OA87.0552/AA87.7162评分复核通过。该LoRA报告未提供训练commit、运行版本或峰值显存，不补造。此次未新运行五项GPU任务，用户重启后产物仍独立保存。

## 2026-10-01 SSH只读补齐原五轮V1产物身份

连接`connect.bjb1.seetacloud.com:50241`，只读JSON、日志、文件列表与torch.save ZIP CRC，无torch导入、模型前向、GPU调用或服务器文件写入。修复后的本机校验函数在内存中对真实文件执行，三个PathVQA身份校验及两套RSVQA完整既有评分校验均通过；这不是新训练或Test评估。

以下路径位于`pathvqa/outputs/visual_selection_prefix/`，均固定`checkpoints/epoch_5`，原V1 method、1864963参数、model seed如下/data42、5epochs、保存3/4/5、batch2/累积16、loss kwargs=False、Accelerate累积1。三个历史运行均明确记录torch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0；原日志确认P20 .3、S8 3e-5、Av10 1e-4，其余条件组1e-4。

| seed | 精确运行目录 | 实际训练提交 | 训练秒 | 报告峰值显存bytes | 已有Validation Overall |
|---|---|---|---:|---:|---:|
|44|pathvqa_v1_norm_fixed_5ep_seed44_20260926_2|2957ce297637d65a8dcdb5a075c1528072497590|10863.0715|25011373568|59.3386|
|45|pathvqa_v1_norm_fixed_5ep_seed45_20260926|f3b9c3922ec44f3388a43b6fc9cf50d28bf30672|10667.1862|25011373568|58.5077|
|46|pathvqa_v1_norm_fixed_5ep_seed46_20260926|548335ffa98594b478117139c80a5ff128cff8ce|10857.1599|25011373568|58.7314|

明确更正此前seed45/46路径/提交/版本“未知”状态：现由实际产物证实，历史分数不变。seed44前两个无后缀/`_1`目录无train_report，绑定完成的`_2`，不按最新选择。epoch5权重SHA256：44=`18ee0c0f2f3b41f107752b2d160c9c5154bc70b3d8366e5e04a7ba9df8e64aa2`，45=`ec5de2aec77e0783c340207ac51f4ef85f1af7ae5dc5c506170420583efd2796`，46=`ee2915d28a6b8b933cb51fef7682c4d467fe6b3c2651466d4db761b9704e11da`。入口已改为这三个精确路径，仍执行身份与归一化核验。

RSVQA V1 `rsvqa_v1_norm_fixed_5ep_b4a8_seed44_20260929`报告确认五轮、batch4/累积8、1864963参数，训练提交`1ae3d1362dddd9d9ec02074eb9bb45c24325e03e`、上述相同版本、训练10982.6985秒、报告峰值12892518912bytes。既有Test10004题/100图、OA85.0360/AA86.0812评分复核通过。RSVQA LoRA `rsvqa_lr_lora_full_model_attention_r8_b4a8_seed44_20260929`报告确认三轮/r8 alpha16/7077888参数/batch4累积8/归一化False、训练12284.0502秒；既有Test OA87.0552/AA87.7162评分复核通过。该LoRA报告未提供训练commit、运行版本或峰值显存，不补造。此次未新运行五项GPU任务，用户重启后产物仍独立保存。

## 2026-10-01 五项Test串行首次预检失败（未启动GPU）

用户执行7461be1五项入口，在绑定原五轮PathVQA V1 seed44时因旧`train_report.json`不含`visual_prompt_mode`而停止。失败目录`/root/autodl-tmp/Qwen3-VL-modify-test/outputs/five_task_test_suite/precheck_failed_20261001_133024_919726`；目标产物`pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2/checkpoints/epoch_5`。尚未执行三seed Test或RSVQA两项训练，无新Overall/分项；运行库版本未回传。属于预检schema兼容错误，不是checkpoint已证实错误或模型失败。

已核对最初五轮实现c270146：报告没有视觉mode/Av10 LR/optimizer group rates。此次修复仅兼容元数据：缺mode时要求报告与checkpoint均明确原V1 method且无消融，继续严格核对布局、参数、seed、五轮及归一化；分组LR优先原报告，其次原`train.log`，再按报告完整训练commit读取当时V1Trainer字面量，不用当前HEAD/默认值推定。证据缺失、冲突或配置不符仍停。CPU回归覆盖旧schema接受、错误Av10 LR拒绝及无LR证据拒绝；任务顺序、训练和生成协议不变。等待用户亲自重启入口。

## 2026-10-01 五项Test串行首次预检失败（未启动GPU）

用户执行7461be1五项入口，在绑定原五轮PathVQA V1 seed44时因旧`train_report.json`不含`visual_prompt_mode`而停止。失败目录`/root/autodl-tmp/Qwen3-VL-modify-test/outputs/five_task_test_suite/precheck_failed_20261001_133024_919726`；目标产物`pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2/checkpoints/epoch_5`。尚未执行三seed Test或RSVQA两项训练，无新Overall/分项；运行库版本未回传。属于预检schema兼容错误，不是checkpoint已证实错误或模型失败。

已核对最初五轮实现c270146：报告没有视觉mode/Av10 LR/optimizer group rates。此次修复仅兼容元数据：缺mode时要求报告与checkpoint均明确原V1 method且无消融，继续严格核对布局、参数、seed、五轮及归一化；分组LR优先原报告，其次原`train.log`，再按报告完整训练commit读取当时V1Trainer字面量，不用当前HEAD/默认值推定。证据缺失、冲突或配置不符仍停。CPU回归覆盖旧schema接受、错误Av10 LR拒绝及无LR证据拒绝；任务顺序、训练和生成协议不变。等待用户亲自重启入口。

## 2026-09-30 PathVQA LoRA-r2 batch2 OOM及batch1重跑准备

- 失败实验：`pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_seed44`，PathVQA，model seed44/data seed42，标准全attention rank2/alpha4，batch2/累积16，计划五轮固定epoch5 Validation。用户回传训练在837/3075步（27%，耗时1:06:37）反向CUDA OOM：申请1.19 GiB，GPU总31.36 GiB、空闲337 MiB，进程31.02 GiB，PyTorch已分配27.84 GiB、保留未分配2.53 GiB。最后已记录epoch1.334；无完成的epoch5成绩，Overall及分项均缺失，不能作为性能结果。
- 用户确认失败运行是重启后的`_1`后缀。输出父目录`pathvqa/outputs/lora/`；精确日期及完整运行路径、实际执行提交与库版本未回传，不自行补造。对应本地准备代码提交65967a5，不等同已核实服务器执行提交。不能仅凭该日志判定泄漏、异常样本或碎片化。
- 用户授权唯一配置变更为microbatch1/累积32，等效batch32；新实验`pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44`，从头训练，独立目录，原产物保留。其余结构、初始化、学习率、优化器、归一化、五轮及固定epoch5协议不变。不额外加allocator配置或checkpointing。等效batch相同不保证随机性及等权microbatch目标下逐步轨迹完全相同。
- 状态：代码准备，尚未运行；GPU由用户执行，不自动关机、不自动重试。结果回传后补记精确路径、成绩、配对统计与成本。

## 2026-09-13 correction: private electrical evaluation denominator

The user, who authored the evaluator, confirmed that all methods automatically received credit for the same 18 samples with missing image-file associations. The previously recorded 70.06 / 70.88 / 71.91 percentages used all 972 samples, including these automatic credits; they are not accuracies over the 954 evaluated samples. Historical entries below are preserved.

- Corrected protocol: exclude those same 18 samples from both numerator and denominator. No model retraining or new inference was performed.
- electrical_static_prompt_p20_seed47_20260911: the reported rounded total corresponds to681/972; excluding automatic credits gives **663/954 = 69.4968553459%**, reported as **69.50%**.
- electrical_cocoop_style_p20_h160_seed47_20260911:689/972 becomes **671/954 = 70.3354297694%**, reported as **70.34%**.
- electrical_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed47_20260911_1:699/972 becomes **681/954 = 71.3836477987%**, reported as **71.38%**.
- These integer totals are reconstructed from the previously reported two-decimal scores with denominator972 and the user-confirmed scoring rule; raw per-sample files were not reread in this correction. Model seed47, data seed42, epoch3 checkpoints and output paths under the existing experiment roots remain unchanged.
- QDPT gains over Static and CoCoOp-style are **18/954*100 = 1.8867924528** and **10/954*100 = 1.0482180294** percentage points, reported as **1.89 / 1.05**.
- The user verified the electrical Full-Attention LoRA-r8 evaluator output as `rank8: score=70.96 evaluated=954`, with **677/954 = 70.9643605870%**, reported as **70.96%**. The historical70.69% and the tentative687/972 count came from a transposed or mixed-denominator record and are superseded by this direct954-sample result.
- QDPT exceeds Full-Attention LoRA-r8 by **4/954*100 = 0.4192872117** percentage points, reported as **0.42**. The corrected single-seed ordering on the954 valid samples is Static69.50 < CoCoOp-style70.34 < Full-Attention LoRA-r8 70.96 < QDPT71.38.
- No answer-type breakdown, paired analysis or new seed results were supplied.

## 2026-09-13 paper configuration clarification supplied by user

- Backbone: Qwen3-VL-4B-Instruct, approximately4.445B parameters,24 visual Transformer blocks at width1024 and36 language blocks at width2560. Original backbone parameters frozen; BF16; native AutoProcessor dynamic resolution with no method-specific fixed image resolution or visual-token cap; text maximum length2048.
- QDPT: AdamW betas0.9/0.999, epsilon1e-8, all weight decay0,3% linear warmup then linear decay, max gradient norm1.0, no gradient checkpointing;3 epochs, microbatch2 and accumulation16. Public multi-seed runs use44/45/46, data seed42; public single-run controls use44; private electrical runs retain47. Use epoch3, not validation-best epoch.
- Learning rates: P20 and A_t10 at0.3; S8 at3e-5; A_v10 and the question-guided generator, attention and output projection at1e-4. Paper groups S8 and A_v10 as18 static visual Prompt tokens with18,432 parameters. The other generator modules total7,709,952; P20+A_t10 total76,800; overall7,805,184. Implementation grouping S8=8,192 and A_v10+generator=7,720,192 is an equivalent accounting.
- Output head: LayerNorm at width768, biased768-to768 Linear, tanh-approximate GELU, biased768-to2560 Linear. Last-layer weight and bias initialized to zero; output added to A_t10. Visual read/Prompt site is before zero-index17, the18th block; LLM order is [P20; Visual; A_t+g(Z)10; Question].
- Full-Attention LoRA: rank8, alpha16, dropout0.05, no bias training or DoRA;24 visual layers qkv/proj and36 language layers q/k/v/o,192 Linear targets,7,077,888 trainable parameters. AdamW LR1e-4, weight decay0,3% warmup and linear decay, microbatch1 and accumulation32; same public seed, precision, epoch and checkpoint protocol.
- Static P20:51,200 parameters, LR0.3. CoCoOp-style P20/H160:873,120 parameters, Prompt/Meta-Net LR0.3/3e-4. Architecture/hyperparameters selected on PathVQA Validation and fixed for final multi-seed and cross-dataset runs. Generation length and decoding arguments remain to be documented.

Last updated: 2026-08-31

This file is the persistent source of truth for completed experiments. Results are recorded from official SLAKE evaluation output or diagnostics supplied during development. Unless noted otherwise, SLAKE evaluation contains 2,094 test questions, uses all languages, and reports percentages.

## Recording Rules

- Record every completed experiment, including failed runs and negative results.
- Record the exact experiment name, seed, architecture/loss change, Overall score, and available breakdowns.
- Do not overwrite historical rows. Add corrections with an explanation.
- Multi-seed summaries use the sample standard deviation.
- A result without an exact log or official score must be marked approximate or unresolved.
- Update this file in the same change that adds or completes an experiment, then commit and push it.

## Current Snapshot

- Current PathVQA matched comparison: frozen Base **34.77**, minimal last8-layer MMRL + Relation0.05 seed44 **49.41**, and last8-layer Visual Attention LoRA-r128 seed44 **53.85**. The scope-matched LoRA lead is **4.44** points; all24-layer LoRA remains a broader upper bound at **55.45**.
- Strongest overall result: static Prompt Tuning, seed44, **74.40** with only 51,200 trainable parameters.
- Strongest MMRL single run: 128-slot Attention Pooling + shared CA + same-initialized layer MLPs + Relation 0.05, seed44, **73.93**.
- Final Mean Pooling MMRL across seeds44-47: **72.98 +/- 0.49**.
- 128-slot Attention Pooling MMRL across seeds44-47: **72.94 +/- 0.66**.
- Mean Pooling and 128-slot Attention Pooling are statistically tied on SLAKE; Mean Pooling is simpler and less variable.
- The current naive parameter-matched Concat-MLP replacement scores **69.25**, but its joint normalization is confounded by Query/Memory scale mismatch. A separately normalized Concat-MLP is pending before claiming CA is irreplaceable.
- A second dataset is required before making general claims about Pooling, Relation, or Prompt Tuning.

## Current MMRL Definition

The current clean MMRL candidate is:

```text
Shared low-dimensional latent table S
  -> 8 same-initialized but independently optimized layer MLPs
  -> layer-specific base prompts Q

Visual Mean Pooling + Text Mean Pooling
  -> K/V memory

Shared residual Cross-Attention(Q, K, V)
  -> 40 sample-conditioned prompts per selected ViT layer
  -> replacement/refresh injection into ViT layers 17-24
```

Default training configuration: seed varies, data seed42, 3 Stage-3 epochs, effective batch32, Relation weight0.05, open/full CE training gate.

## Formal Multi-Seed Results

### Pooling Comparison

| Method | Seed44 | Seed45 | Seed46 | Seed47 | Mean +/- SD |
|---|---:|---:|---:|---:|---:|
| 128-slot Attention Pooling + Relation | 73.93 | 72.64 | 72.54 | 72.64 | 72.94 +/- 0.66 |
| Mean Pooling + Relation | 72.97 | 73.26 | 73.40 | 72.30 | 72.98 +/- 0.49 |

Conclusion: the mean difference is only +0.04 for Mean Pooling. The learned 128-slot pooling does not provide a stable SLAKE gain.

### Full-CE Relation Weighting

| Relation weighting | Seed44 | Seed45 | Seed46 | Seed47 | Mean +/- SD |
|---|---:|---:|---:|---:|---:|
| Uniform | 73.93 | 72.64 | 72.54 | 72.64 | 72.94 +/- 0.66 |
| Alpha-weighted | - | 72.73 | 72.11 | 73.02 | 72.62 +/- 0.46 over 3 seeds |

Conclusion: alpha-weighting did not improve the multi-seed mean.

### Gate Strategy

| Gate strategy | Seed44 | Seed45 | Seed46 | Seed47 | Mean +/- SD |
|---|---:|---:|---:|---:|---:|
| Hard Concrete | 72.59 | 71.30 | 73.50 | 73.45 | 72.71 +/- 1.03 |
| Alpha probability | 73.59 | 72.92 | 73.02 | 72.16 | 72.92 +/- 0.59 |

Conclusion: alpha probability reduced variance but did not materially improve the mean.

## Cross-Attention Delta Scale

### Seed44

| Scale | Overall | Delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 45.32 | -28.08 | 34.83 | 46.85 | 54.90 | 38.95 | 53.44 | 36.98 |
| 0.5 | 72.68 | -0.72 | 55.81 | 75.15 | 80.74 | 67.33 | 75.21 | 70.09 |
| 1.0 | 73.40 | +0.00 | 56.55 | 75.86 | 80.26 | 68.84 | 76.06 | 70.67 |

### Seed45

| Scale | Overall | Delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 45.46 | -26.65 | 35.58 | 46.91 | 55.14 | 39.03 | 53.44 | 37.27 |
| 0.5 | 71.20 | -0.91 | 53.56 | 73.78 | 81.22 | 64.55 | 75.78 | 66.51 |
| 1.0 | 72.11 | +0.00 | 55.43 | 74.55 | 82.66 | 65.10 | 76.44 | 67.67 |

Conclusion: the learned CA delta is essential; removing it loses about 27 points. Half scale retains most but not all performance.

## Query-Generator Swap

| Composition | Overall | Recipient delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q44 + Rest44 | 73.40 | +0.00 | 56.55 | 75.86 | 80.26 | 68.84 | 76.06 | 70.67 |
| Q45 + Rest45 | 72.11 | +0.00 | 55.43 | 74.55 | 82.66 | 65.10 | 76.44 | 67.67 |
| Q44 + Rest45 | 53.39 | -18.72 | 41.20 | 55.17 | 58.37 | 50.08 | 62.58 | 43.95 |
| Q45 + Rest44 | 53.77 | -19.63 | 41.57 | 55.56 | 56.82 | 51.75 | 64.75 | 42.50 |

Conclusion: the query generator and recipient parameters co-adapt strongly and cannot be swapped independently.

## Architecture Experiments

| Experiment | Seed | Overall | Status / conclusion |
|---|---:|---:|---|
| `slake_mmrl_layer_mlp_full_ca_relation0050_repro3_seed44_20260817_1` | 44 | 73.40 | Historical Layer-MLP full-CA baseline |
| Layer-MLP full-CA counterpart | 45 | 72.11 | Historical seed45 counterpart |
| `slake_mmrl_layer_mlp_same_init_relation0050_seed44_20260819` | 44 | 73.93 | Same-origin MLP initialization; strongest MMRL single run |
| `slake_mmrl_layer_mlp_deepstack_relation0050_seed44_20260819` | 44 | 68.53 | DeepStack residual failed; delta/original ratio reached about0.22 late |
| `slake_mmrl_layer_mlp_cross_relation0010_relation0050_seed44_20260819` | 44 | 68.29 | Mixed old Relation + Cross-Relation 0.01 failed |
| `slake_mmrl_layer_mlp_current_control_relation0050_seed44_20260819` | 44 | 71.87 | Code-state control; exposed training trajectory instability |
| `slake_mmrl_layer_mlp_same_init_cross_relation_only0050_seed44_20260819` | 44 | 72.68 | Cross-Relation only; below 73.93 reference |
| `slake_mmrl_layer_mlp_same_init_reverse_assignment_relation0050_seed44_20260819` | 44 | 70.96 | Reverse assignment attention failed |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_static_kv_relation0050_seed44_20260820` | 44 | 61.70 | Dynamic serial query; initialization scale was invalid |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 67.38 | Zero-gated repair improved but remained poor |
| `slake_mmrl_layer_mlp_same_init_pooled_query_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 65.66 | Pooled query attention was nearly uniform |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_competitive_visual_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 67.19 | Competitive visual assignment remained nearly uniform |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_dual_softmax_visual_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 67.57 | Dual-softmax repair did not recover baseline |

### Detailed Relation / Assignment Results

| Experiment | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current control | 71.87 | 79.55 | 66.77 | 52.43 | 74.71 | 74.27 | 69.41 |
| Cross-Relation only | 72.68 | 78.59 | 68.76 | 54.68 | 75.31 | 75.31 | 69.99 |
| Reverse assignment | 70.96 | 77.87 | 66.38 | 52.06 | 73.73 | 74.27 | 67.57 |

Reverse-assignment diagnostics: normalized entropy about0.990, indicating diffuse assignment; slot mass became imbalanced despite high entropy.

## Gate and Reproducibility Experiments

| Experiment | Seed | Overall | Notes |
|---|---:|---:|---|
| Same-init original run | 44 | 73.93 | Stage-1 hidden pooling delta0.7061 |
| Same-init repro4 | 44 | 69.29 | Same nominal seed/config; Stage-1 hidden pooling delta0.2922 |
| No gate (`G=1`) | 44 | 71.97 | Gate ablated; changed optimization trajectory |
| Gated rerun | 44 | 73.07 | `G_mean` about0.988; confirms rerun variability |
| Mislaunched ordinary Layer-MLP run | 44 | 73.69 | Output name lacked `alpha_prob`; not valid as alpha experiment |

Conclusion: nominally decoupled gate and Relation components alter shared optimization trajectories. Same seed/config was not perfectly reproducible in older code states, so later formal results use explicit open/full-CE configuration and multi-seed reporting.

## Final SLAKE Ablations

### Attention-Pooling Configuration

| Experiment | Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Relation | 44 | 72.83 | 79.07 | 68.68 | 54.31 | 75.53 | 75.40 | 70.18 |
| No Relation | 45 | 72.92 | 81.58 | 67.17 | 53.56 | 75.75 | 75.21 | 70.57 |
| Independent MLP initialization | 44 | 72.68 | 80.98 | 67.17 | 54.31 | 75.37 | 75.59 | 69.70 |
| Independent MLP initialization | 45 | 72.83 | 82.06 | 66.69 | 53.56 | 75.64 | 75.02 | 70.57 |
| Static query / no dynamic fusion | 45 | 56.69 | 63.76 | 51.99 | 40.82 | 59.00 | 64.66 | 48.50 |

Two-seed summaries: No Relation **72.88 +/- 0.06**; independent initialization **72.76 +/- 0.11**. The static-query collapse proves that sample-conditioned fusion is necessary.

### Mean-Pooling Configuration

| Experiment | Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean Pooling + Relation | 44 | 72.97 | - | - | - | - | - | - |
| Mean Pooling + Relation | 45 | 73.26 | 82.78 | 66.93 | 52.81 | 76.25 | 75.12 | 71.35 |
| Mean Pooling + Relation | 46 | 73.40 | 82.42 | 67.41 | 53.56 | 76.30 | 75.49 | 71.25 |
| Mean Pooling + Relation | 47 | 72.30 | 79.67 | 67.41 | 55.81 | 74.71 | 75.21 | 69.31 |
| Mean Pooling, no Relation | 45 | 73.30 | 81.46 | 67.89 | 52.43 | 76.35 | 76.34 | 70.18 |
| Mean Pooling, independent initialization | 45 | 72.59 | 81.58 | 66.61 | 55.81 | 75.04 | 75.49 | 69.60 |

Conclusion: Relation has no visible Overall gain on Mean Pooling at seed45 (+0.04 when removed). Same initialization improves Overall by0.67 at seed45, although KVQA moves in the opposite direction.

## Pooling Redesign Attempts

| Experiment | Seed | Overall | Key diagnostics / conclusion |
|---|---:|---:|---|
| Text-guided visual slots8, selection-only | 45 | 73.16 | Attention entropy0.964; slot pair cosine0.985; soft slot collapse |
| Text-guided standard residual CA slots8 + Relation | 45 | 71.11 | Entropy0.898; slot cosine0.476; slots became diverse but accuracy fell |
| Text-guided standard residual CA slots8, no Relation | 45 | 71.49 | Relation removal recovered only0.38 |
| Balanced text/visual fusion slots8 + Relation | 45 | 70.49 | Entropy0.982; slot cosine0.996; equal fusion collapsed slots again |

Conclusion: under current CE supervision, more elaborate learned pooling either remains diffuse or becomes diverse without improving task accuracy.

## Fusion Replacement

| Fusion | Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared CA + Mean Pooling | 45 | 73.26 | 82.78 | 66.93 | 52.81 | 76.25 | 75.12 | 71.35 |
| Naive equal-parameter Concat-MLP + Mean Pooling | 45 | 69.25 | 76.79 | 64.23 | 52.81 | 71.65 | 73.52 | 64.86 |

Concat-MLP audit:

- Fusion parameters: 4,202,496, exactly matching CA.
- Total MMRL parameters: 20,506,624.
- CE decreased normally and gradients were active; this was not a dead module.
- Query norm was about0.67 while visual/text memory norms were about32/26.
- The implementation used one joint `LayerNorm(3D)`, whereas CA separately normalizes Q and memory.
- Therefore the 4.01-point gap currently proves naive concatenation is poor, but does not yet isolate attention from normalization. The fair follow-up is `Concat([LN(Q), LN(V), LN(T)]) -> MLP`, which preserves the exact parameter budget.

## Static Prompt Tuning Baseline

Experiment: `slake_prompt_tuning_len20_seed44_20260825_1`

| Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | 74.40 | 80.62 | 70.27 | 62.55 | 76.14 | 75.40 | 73.38 |

Training audit:

- Prompt length20, LLM hidden size2,560.
- Exactly 51,200 trainable parameters.
- 3 epochs, seed44, data seed42, learning rate0.3.
- Runtime4,320 seconds; train loss3.723.
- About924 optimization steps, matching the MMRL three-epoch schedule.
- No known train/test leakage; treat the result as valid unless a later audit contradicts it.

Interpretation: Prompt Tuning matches MMRL on VQA but gains heavily on KVQA and OPEN questions. It directly conditions the LLM's knowledge and answer space, while MMRL primarily specializes the visual path. This is now the strongest SLAKE baseline and must be included in future comparisons.

## PathVQA Results

### 2026-08-26 - pathvqa_mmrl_layer_mlp_same_init_full_ce_uniform_relation0050_seed44_20260826

- Commit/config: PathVQA MMRL pipeline at commit `bc84ee3`; 128-slot Attention Pooling, shared Cross-Attention, same-initialized layer MLPs, 40 Rep tokens in visual layers17-24, `open_full_ce`, uniform Relation weight0.05.
- Dataset and split: PathVQA official test, all6,719 questions, 858 image clusters; train contains19,654 samples.
- Seed / data seed: 44 / 42.
- Controlled change: first formal transfer of the frozen-LLM MMRL method from SLAKE to PathVQA.
- Overall: **47.30**; image-clustered 95% bootstrap CI **[46.08, 48.49]**.
- Yes/No / Free-form: **82.57 / 11.97**.
- Per question type: how10.79, other0.00, what6.82, when0.00, where46.40, why0.00, yes/no82.57; Free-form question-type macro10.67.
- Stage-3 trainable MMRL parameters:20,998,400; compact checkpoint stores24,288,513 parameters including the trained Stage-1 Gate path.
- Evaluation timing: TTFT mean0.0577s, TPOT0.02095s/token,47.74 decode tokens/s, request mean0.1103s.
- Diagnostics: 1,845 steps, 8 windows, no malformed rows; CE mean0.8668 and first/last windows1.5247/0.7966; Cross-Attention grad norm mean0.7225; `delta_to_org_ratio`0.6176; Cross-Attention delta/base ratio6.781; text/visual pooling grad norms0.4438/0.1285; scaled Relation loss0.001577; `G=1` throughout Stage3 as required by `open_full_ce`.
- Conclusion: training and gradient flow were active, but performance is sharply split between Yes/No and open answers. The result is poor as an absolute score, yet attribution is deferred until the matched Visual LoRA-r128 and frozen Base evaluations finish; if all three have low Free-form accuracy, the bottleneck is likely the frozen LLM/answer-space interface rather than an MMRL-only failure.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_layer_mlp_same_init_full_ce_uniform_relation0050_seed44_20260826`.

### 2026-08-26 - pathvqa_lora_visual_all_attention_r128_seed44_20260826

- Commit/config: commit `bc84ee3`; LoRA on `qkv/proj` Attention linears in all24 visual layers, rank128, alpha256, dropout0.05; visual MLP and full LLM frozen.
- Dataset and split: PathVQA official train19,654; checkpoint selection on validation6,259; one final evaluation on test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: standard visual-only Attention LoRA baseline against frozen-LLM MMRL; no language-layer LoRA.
- Validation Overall by epoch: **48.94, 54.91, 55.70**; earliest-best selection chose epoch3. Validation Yes/No by epoch:81.98,86.69,85.82; Free-form:15.99,23.23,25.65.
- Test Overall: **55.45**; image-clustered 95% bootstrap CI **[53.99, 56.99]**.
- Test Yes/No / Free-form: **86.29 / 24.58**.
- Per question type: how5.76, other5.56, what20.50, when0.00, where58.93, why0.00, yes/no86.29; Free-form question-type macro15.12.
- Trainable parameters:18,874,368 of4,456,690,176 total, ratio0.4235%.
- Training: 3 epochs, LR1e-4, micro-batch1, gradient accumulation32; runtime9,882.8s, loss0.7361.
- Evaluation timing: TTFT mean0.0523s, TPOT0.01991s/token,50.23 decode tokens/s, request mean0.1025s.
- Conclusion: LoRA exceeds MMRL by8.16 Overall,3.72 Yes/No, and12.60 Free-form points, so visual specialization is learnable and MMRL transfers poorly relative to this standard baseline. Absolute performance remains low and epoch3 was still improving, but no rank/MLP/epoch expansion is scheduled before the frozen Base result establishes the real gain over the original model.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_visual_all_attention_r128_seed44_20260826`; selected checkpoint `checkpoints/epoch_3`.

Correction (2026-08-26): the intended scope-matched LoRA baseline is Attention LoRA-r128 on visual layers17-24, because MMRL also modifies only those last8 visual layers. This completed LoRA run targets all24 visual layers and must therefore be treated as a broader-scope upper-bound baseline, not as evidence that LoRA is intrinsically8.16 points better under matched adaptation scope. A last8-layer PathVQA LoRA-r128 run is required for the formal comparison.

### 2026-08-26 - pathvqa_base_20260826

- Commit/config: commit `bc84ee3`; frozen original `/root/autodl-tmp/model`, no checkpoint and no training; generation/evaluation settings exactly match MMRL and LoRA.
- Dataset and split: PathVQA official test, all6,719 questions and858 image clusters.
- Seed: not applicable; deterministic greedy decoding with temperature0.0.
- Controlled change: unadapted Base control for the first PathVQA method comparison.
- Overall: **34.77**; image-clustered 95% bootstrap CI **[33.79, 35.81]**.
- Yes/No / Free-form: **67.34 / 2.14**.
- Per question type: how2.16, other0.00, what1.35, when0.00, where7.42, why0.00, yes/no67.34; Free-form question-type macro1.82.
- Evaluation timing: TTFT mean0.0484s, TPOT0.02166s/token,46.18 decode tokens/s, request mean0.1171s.
- Three-way deltas: MMRL versus Base is **+12.53 Overall, +15.23 Yes/No, +9.83 Free-form**; LoRA versus Base is **+20.69 Overall, +18.95 Yes/No, +22.43 Free-form**; LoRA versus MMRL is **+8.16 Overall**.
- Conclusion: PathVQA is genuinely difficult for the original model, especially under normalized exact-match for open answers. MMRL is not a failed or dead adaptation: it delivers a large gain over Base. However, standard visual-only LoRA learns the task substantially better, especially on Free-form answers, so the current MMRL cannot claim superior effectiveness on the primary dataset. Qualitative Base outputs also show plausible pathology phrases that may be semantically related but fail the single-reference exact match; semantic scoring or a blinded error sample should supplement, not replace, the official metric.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/base/pathvqa_base_20260826`.

### 2026-08-27 - pathvqa_lora_visual_last8_attention_r128_seed44_20260826

- Commit/config: commit `96f0550`; LoRA-r128 on visual Attention `qkv/proj` in layers17-24 only (0-based16-23), alpha256, dropout0.05; visual MLP and LLM frozen.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: scope-matched visual-only LoRA baseline using exactly the same last8 visual layers affected by MMRL.
- Validation Overall by epoch: **49.5606, 52.8998, 53.8904**; epoch3 selected.
- Test Overall: **53.8473**; image-clustered 95% bootstrap CI **[52.3630, 55.4189]**.
- Yes/No / Free-form: **84.2356 / 23.4138**.
- Per question type: how5.7554, other5.5556, what18.7523, when0.0000, where61.0209, why0.0000, yes/no84.2356; Free-form question-type macro15.1807.
- Trainable parameters: **6,291,456**; target audit selected0-based layers16-23 and exactly16 `qkv/proj` modules.
- Training: 3 epochs, LR1e-4, micro-batch1, gradient accumulation32; runtime8,772.4s; train loss0.7768.
- Conclusion: last8 LoRA retains most of the all24-layer LoRA result:53.85 versus55.45, only1.60 points lower while using one-third as many LoRA parameters. It is the valid scope-matched visual-only baseline.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_visual_last8_attention_r128_seed44_20260826`; selected checkpoint `checkpoints/epoch_3`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_relation0_seed44_20260826

- Commit/config: commit `96f0550`; 40x1024 shared full-dimensional `S` directly enters one shared8-head residual Cross-Attention; Mean Pooling supplies one visual and one text memory token; no layer MLP projectors; layer embeddings retained; insertion into visual layers17-24; `open_full_ce`; Relation0.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: minimal MMRL architecture and Relation-off half of an exact shared-Stage1 comparison.
- Validation Overall by epoch: **46.2853, 46.8605, 46.6049**; epoch2 selected.
- Test Overall: **46.8373**; image-clustered 95% bootstrap CI **[45.5662, 48.2917]**.
- Yes/No / Free-form: **79.1196 / 14.5070**.
- Yes/No class audit:1,816 `yes` questions at69.1079% and1,546 `no` questions at90.8797%; without Relation the model remains strongly biased toward `no`.
- Per question type: how8.6331, other0.0000, what10.4706, when0.0000, where43.6195, why0.0000, yes/no79.1196; Free-form question-type macro10.4539.
- Trainable MMRL parameters: **7,927,808** exactly: shared S40,960; layer embeddings8,192; projectors0; visual Mean Pooling1,051,648; text Mean Pooling2,624,512; shared CA4,202,496.
- Stage3 runtime/loss:6,616s /0.8948.
- Diagnostics:1,845 steps,8 windows, no malformed rows; CE mean0.8919; CA grad norm0.7160; `delta_to_org_ratio`0.6680; CA delta/base ratio5.5979; text/visual pooling grad norms0.8021/0.2468; Relation loss exactly0.
- Conclusion: the minimal architecture remains a real specialist at+12.07 points over Base, but without Relation it does not exceed the earlier20.998M-parameter MMRL result.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_relation0_seed44_20260826`; selected checkpoint `checkpoints/stage3_epoch_2`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_relation0050_seed44_20260826

- Commit/config: identical to the preceding minimal MMRL except uniform Relation weight0.05. It strictly reloads the same Stage1 checkpoint and untouched initial MMRL state with SHA-256 `a2cef33f65e3e837e99a0d188fd4186612f331d491b782f31314a5aa1bc1a093`.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: Relation0.05 versus0 under an exact shared-Stage1, matched-initialization comparison.
- Validation Overall by epoch: **45.4545, 49.5447, 48.7618**; epoch2 selected.
- Test Overall: **49.4121**; image-clustered 95% bootstrap CI **[48.1175, 50.7317]**.
- Yes/No / Free-form: **82.5996 / 16.1752**.
- Yes/No class audit:1,816 `yes` questions at79.6806% and1,546 `no` questions at86.0285%. Relative to the paired Relation0 run, Relation0.05 changes `yes` by+10.5727 and `no` by-4.8512 points, trading a modest loss on `no` for a much larger recovery on `yes`.
- Per question type: how9.3525, other0.0000, what11.2003, when0.0000, where51.7401, why0.0000, yes/no82.5996; Free-form question-type macro12.0488.
- Trainable MMRL parameters: **7,927,808**; parameter audit passed exactly.
- Stage3 runtime/loss:6,617s /0.8953.
- Diagnostics:1,845 steps,8 windows, no malformed rows; CE mean0.8915; CA grad norm0.6972; `delta_to_org_ratio`0.6873; CA delta/base ratio5.4629; text/visual pooling grad norms0.7683/0.2474; scaled Relation loss0.001819.
- Paired Relation effect: **+2.5748 Overall**, +3.4801 Yes/No, +1.6682 Free-form. Discordant correctness counts are284 Relation0-only versus457 Relation0.05-only (`McNemar p=2.20e-10`); image-clustered bootstrap 95% CI for the Overall difference is **[+1.7217, +3.4544]**.
- Matched LoRA comparison: last8 LoRA is+4.4352 Overall, +1.6359 Yes/No, and+7.2386 Free-form; clustered bootstrap 95% CI for its Overall lead is **[+3.3462, +5.5093]**.
- Conclusion: Relation now has a clear positive PathVQA effect under a clean controlled comparison, not a Gate/Stage1 trajectory confound. The minimal MMRL also improves over the earlier47.30 MMRL while cutting Stage3 MMRL parameters from20.998M to7.928M, although epoch-selection and architecture both changed, so that cross-run gain is descriptive rather than a pure ablation. Last8 LoRA remains stronger, primarily on Free-form answers.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_relation0050_seed44_20260826`; selected checkpoint `checkpoints/stage3_epoch_2`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_grid5x8_relation0050_seed44_20260827

- Commit/config: commit `7b7bc44`; identical to the minimal shared-S Mean-Pooling MMRL + Relation0.05 run except that the 40 Rep Tokens use fixed cell-center RoPE coordinates on a 5x8 image grid instead of sharing the origin.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: Rep Token RoPE positions only; fixed coordinates, no learnable position parameters, and the same Stage1 checkpoint SHA-256 `a2cef33f65e3e837e99a0d188fd4186612f331d491b782f31314a5aa1bc1a093`.
- Validation Overall by epoch: **46.52, 45.90, 47.26**; epoch3 selected.
- Test Overall: **47.67**.
- Yes/No / Free-form: **81.02 / 14.27**.
- Trainable MMRL parameters: **7,927,808** exactly; unchanged from the origin-position control.
- Stage3 runtime/loss:6,608s /0.9042.
- Position audit: `grid_5x8`,40 slots and40 unique positions per sequence; the sampled first sequence spans `[2,1]` to `[18,28]`.
- Diagnostics:1,845 steps,8 windows, no malformed rows; CE mean0.9012; CA grad norm0.7757; `delta_to_org_ratio`0.6597; CA delta/base ratio5.8809; text/visual pooling grad norms0.8550/0.2431; scaled Relation loss0.001710.
- Matched effect versus origin-position Relation0.05: **-1.74 Overall, -1.58 Yes/No, and -1.91 Free-form**.
- Conclusion: fixed spatial RoPE is active and optimization remains healthy, but it consistently harms both answer categories. The shared Rep Tokens are semantic prompts rather than tokens aligned to fixed image patches; forcing distinct spatial phases introduces a false correspondence and breaks their useful shared-origin symmetry. Reject this direction and do not add learnable coordinates without a new token-to-region alignment mechanism.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_grid5x8_relation0050_seed44_20260827`; selected checkpoint `checkpoints/stage3_epoch_3`.

### 2026-08-27 - pathvqa_prompt_tuning_len20_seed44_20260827

- Commit/config: commit `722a65f`; classic static Prompt Tuning with20 learned prefix embeddings of width2,560; the complete Qwen3-VL backbone is frozen.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: formal frozen-backbone static Prompt Tuning baseline using the same PathVQA templates, evaluator, generation settings, and validation-selected test protocol as MMRL and LoRA.
- Validation Overall by epoch: **52.71, 52.76, 54.87**; epoch3 selected.
- Test Overall: **55.8268**; image-clustered 95% bootstrap CI **[54.4524, 57.1496]**.
- Yes/No / Free-form: **90.3331 / 21.2690**.
- Per question type: how7.9137, other22.2222, what16.4903, when0.0000, where57.3086, why0.0000, yes/no90.3331; Free-form question-type macro17.3225.
- Yes/No class audit:1,816 `yes` questions at91.2996% and1,546 `no` questions at89.1979%; the aggregate gain is not a one-class collapse.
- Trainable parameters: **51,200** exactly (`20 x 2,560`), versus7.928M for minimal MMRL and6.291M for last8 Attention LoRA-r128.
- Training:3 epochs, LR0.3, micro-batch2, gradient accumulation16; runtime5,969.6s; reported train loss12.0592.
- Evaluation timing: TTFT mean0.0474s, weighted TPOT0.01959s/token,51.05 decode tokens/s, request mean0.0989s.
- Deltas versus Base: **+21.0568 Overall, +22.9931 Yes/No, +19.1290 Free-form**.
- Deltas versus minimal MMRL+Relation0.05: **+6.4147 Overall, +7.7335 Yes/No, +5.0938 Free-form**.
- Deltas versus last8 Attention LoRA-r128: **+1.9795 Overall, +6.0975 Yes/No, but -2.1448 Free-form**.
- Conclusion: static Prompt Tuning is the current best PathVQA Overall result and demonstrates that the dominant adaptation bottleneck is the frozen LLM context/interface, not only internal ViT specialization. Contrary to the initial prediction, its largest advantage over LoRA is binary-answer calibration; LoRA remains stronger on Free-form. A dynamic multimodal Prompt should therefore be judged on whether it preserves the static prompt's balanced Yes/No performance while recovering the Free-form gap, not merely on aggregate Overall.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/prompt_tuning/pathvqa_prompt_tuning_len20_seed44_20260827`; selected checkpoint `checkpoints/epoch_3`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_prompt20_relation0050_seed44_20260827

- Commit/config: commits `d057bf3` through `acf809a`; jointly trains the exact7,927,808-parameter minimal shared-S Mean-Pooling MMRL with Relation0.05 and a20-token static Soft Prompt. MMRL LR6e-5 retains fixed warmup plus epoch decay; Prompt LR0.3 uses3% warmup plus linear decay. The Prompt prefix is excluded from `mmrl_gating_mask`, so MMRL text pooling sees only the original question context.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42. The MMRL branch reloads the same Stage1 checkpoint used by the minimal controls; the Prompt uses seed44 token-embedding initialization.
- Controlled change: decisive complementarity test adding the independently strong static Prompt20 LLM-context interface to minimal MMRL+Relation0.05 while freezing the vision backbone and LLM weights.
- Validation-selected epoch: epoch3, validation Overall **55.3763**.
- Test Overall: **55.7523**; image-clustered95% bootstrap CI **[54.4270, 57.1364]**.
- Yes/No / Free-form: **88.9352 / 22.5201**.
- Per question type: how12.2302, other27.7778, what17.5848, when0.0000, where58.4687, why0.0000, yes/no88.9352; Free-form question-type macro19.3436.
- Trainable parameters: MMRL7,927,808 + Prompt51,200 = **7,979,008**. The compact checkpoint contains11,269,121 parameters because it additionally serializes frozen Stage1 delta modules.
- Training:3 epochs,1,845 optimizer steps, runtime6,747s, reported train loss0.7651.
- Diagnostics:8 windows, no malformed lines; CE mean0.7547; total MMRL grad norm0.1592; Prompt grad norm0.00894; CA grad norm0.0967; scaled Relation0.000200; `delta_to_org_ratio`0.1634; CA delta/base ratio20.121; Prompt norm grows from214.02 to663.34. Both branches are active, so the tie is not caused by a dead branch.
- Paired behavior versus Static Prompt: predictions are textually identical on4,460/6,719 questions (66.38%), but correctness changes on709 questions. The combination preserves3,394 jointly correct answers, breaks357 Prompt-correct answers, and fixes352 Prompt-wrong answers. By type, Yes/No has173 broken versus126 fixed, while Free-form has184 broken versus226 fixed. A Prompt-or-combination oracle reaches61.07 Overall, showing latent sample-level complementarity that the always-on joint model cannot arbitrate.
- Versus Static Prompt20: **-0.0745 Overall, -1.3979 Yes/No, +1.2511 Free-form**. Versus minimal MMRL+Relation0.05: **+6.3402 Overall, +6.3356 Yes/No, +6.3449 Free-form**. Versus last8 Visual LoRA: **+1.9050 Overall, +4.6996 Yes/No, -0.8937 Free-form**.
- Evaluation timing: TTFT mean0.06061s, weighted TPOT0.02364s/token,42.30 decode tokens/s, request mean0.11123s.
- Conclusion: the combination is statistically and practically tied with Static Prompt alone and fails the pre-registered success condition because Free-form22.52 remains below last8 LoRA23.41. This is not branch death or a trivial no-op: MMRL shifts capacity from binary calibration toward a different set of Free-form answers, but the shared CE objective provides no sample-wise authority/routing mechanism, so fixes and regressions cancel. The dominant PathVQA bottleneck is the LLM context interface; do not spend more budget on an always-on combination of the current internal-ViT MMRL with static Prompt Tuning.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_prompt20_relation0050_seed44_20260827`; selected checkpoint `checkpoints/stage3_epoch_3`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827

- Commit/config: commit `619787d`; frozen Qwen3-VL backbone with 20 jointly trained 2,560-dimensional Static Prompt tokens. The same tokens act as 20 Cross-Attention queries over two sample-conditioned memory tokens: Mean-Pooled image tokens and Mean-Pooled question-only text tokens. The 256-dimensional 8-head CA returns a zero-initialized residual to the same 20-token LLM prefix. No pretrained Prompt checkpoint, Stage1, Gate, or Relation is used.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: replace Static Prompt20 with an equally long jointly trained dynamic multimodal Prompt. The backbone, question template, answer exclusion, generation settings, validation selection, and official normalized exact-match evaluator remain matched to the Static Prompt baseline.
- Validation Overall by epoch: **52.8040, 54.9609, 55.7278**; epoch3 selected.
- Test Overall: **56.5412**; image-clustered95% bootstrap CI for the model score **[55.2460, 57.8331]**.
- Yes/No / Free-form: **89.8275 / 23.2052**.
- Per question type: how9.3525, other38.8889, what17.5848, when0.0000, where64.0371, why4.5455, yes/no89.8275; Free-form question-type macro22.4015.
- Yes/No class audit:1,816 `yes` questions at93.7225% and1,546 `no` questions at85.2523%. Relative to Static Prompt, `yes` changes by+2.4229 and `no` by-3.9457 points, so the small aggregate Yes/No loss hides a calibration shift rather than uniform degradation.
- Trainable parameters: Static Prompt51,200 + Dynamic CA2,634,240 = **2,685,440**. The checkpoint file is10,745,909 bytes.
- Training:3 epochs, Prompt LR0.3, Dynamic CA LR3e-4, shared3% warmup and linear decay; runtime6,051.7s; reported train loss11.4899.
- Initialization audits: dynamic residual maximum absolute value is exactly0 at step0; memory has shape `(batch,2,2560)`; question-only text Memory has zero answer-token overlap.
- Diagnostics:93 rows over steps0-1,840. In epoch1/2/3 windows, normalized two-token attention entropy averages0.08931/0.000506/0.0000995, while visual attention mass averages0.46042/0.45076/0.443754. Over the final30 records, visual mass is0.443754 with std0.0000111 and entropy is0.0001008, indicating an almost fixed hard allocation of the160 query-head pairs between visual and text Memory. The Dynamic residual remains large and sample-varying in norm: epoch3 `delta/base` mean0.4748, range0.3420-0.6459. Dynamic grad norm remains approximately1.0 while Static Prompt grad norm declines to an epoch3 mean0.01849.
- Paired effect versus Static Prompt20: **+0.7144 Overall**, -0.5057 Yes/No, and **+1.9363 Free-form**. Dynamic-only/Static-only correct counts are375/327 Overall (`McNemar p=0.0760`),127/144 Yes/No (`p=0.3311`), and248/183 Free-form (`p=0.00202`). The image-clustered paired95% bootstrap CI for the Overall difference is **[-0.1021,+1.5520]**. Predictions are textually identical on4,393/6,719 questions (65.38%).
- Comparison to last8 Visual LoRA-r128: Dynamic Prompt is+2.6939 Overall and-0.2086 Free-form; it nearly closes LoRA's Free-form advantage while retaining much stronger binary accuracy.
- Evaluation timing: TTFT mean0.049734s, weighted TPOT0.019937s/token,50.157 decode tokens/s, request mean0.095868s.
- Conclusion: this is the strongest PathVQA seed44 point estimate and the first method to improve Static Prompt specifically on Free-form with a significant paired correctness shift. It does not yet prove a statistically stable Overall improvement because the clustered paired CI crosses zero. The near-zero entropy and exactly stable44.375% visual mass show that CA learned a fixed head/slot modality partition, not visibly sample-dependent attention routing. The residual values can still carry sample-conditioned image/text content, so the correct next test is checkpoint-level causal intervention (`delta=0`, shuffled Memory, fixed/mean residual) before adding seeds or claiming dynamic multimodal routing.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827`; selected checkpoint `checkpoints/epoch_3`; test log `eval_test_epoch_3.log`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827__intervention_zero

- Commit/config: commit `26ef735`; inference-only causal intervention on the selected epoch3 Dynamic Prompt checkpoint. The jointly trained20-token Soft Prompt remains unchanged, but the complete sample-conditioned CA residual is forced to exact zero for every test sample.
- Dataset and split: PathVQA test6,719 with858 image clusters; source training seed/data seed44/42.
- Controlled change: Dynamic Prompt residual only; backbone, checkpoint, prefix length, test order, templates, generation, and official evaluator exactly match the normal56.5412 run.
- Overall: **49.2930**; image-clustered95% CI **[48.1936,50.3944]**.
- Yes/No / Free-form: **87.1802 / 11.3494**.
- Yes/No class audit: `yes`90.5837% and `no`83.1824%.
- Per question type: how7.1942, other11.1111, what7.8803, when0.0000, where35.4988, why0.0000, yes/no87.1802; Free-form macro10.2807.
- Intervention audit:6,719/6,719 samples changed; zero warmup samples.
- Paired normal-minus-zero effect: **+7.2481 Overall**, +2.6472 Yes/No, and **+11.8558 Free-form**. Normal-only/zero-only correct counts are604/117 Overall (`McNemar p=6.36e-80`),157/68 Yes/No (`p=2.79e-9`), and447/49 Free-form (`p=1.86e-81`). The image-clustered paired95% CI for the Overall difference is **[+6.3431,+8.1816]**.
- Inference timing:742.5s for6,719 samples; TTFT mean0.048596s, weighted TPOT0.019972s/token, request mean0.101003s.
- Conclusion: the Dynamic residual is indispensable inside the jointly optimized checkpoint, especially for Free-form. This is not an independently trained Static Prompt baseline: the Soft Prompt co-adapted with CA, so the result proves branch necessity and co-adaptation rather than claiming that any Dynamic model must beat a separately optimized Static Prompt by7.25 points.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827/eval_interventions/zero`; log `eval_intervention_zero.log`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827__intervention_mean_residual_lag32

- Commit/config: commit `26ef735`; inference-only intervention using normal sample residuals for the first32 calibration samples, then replacing every subsequent sample residual with their fixed average.
- Dataset and split: PathVQA test6,719 with858 image clusters; source training seed/data seed44/42.
- Controlled change: residual sample dependence only; all weights and evaluation settings match the normal run. The first32 samples are explicitly retained as calibration and excluded in the supplementary clean-effect calculation.
- Overall: **43.1910**; image-clustered95% CI **[42.1834,44.2155]**.
- Yes/No / Free-form: **79.8037 / 6.5237**.
- Yes/No class audit: `yes`78.5793% and `no`81.2419%.
- Per question type: how7.9137, other33.3333, what2.9186, when0.0000, where28.0742, why4.5455, yes/no79.8037; Free-form macro12.7976.
- Intervention audit:6,719 samples seen,6,687 changed,32 calibration samples.
- Paired normal-minus-mean effect: **+13.3502 Overall**, +10.0238 Yes/No, and **+16.6816 Free-form**. Normal-only/mean-only correct counts are1,014/117 Overall (`McNemar p=7.01e-179`),423/86 Yes/No (`p=1.81e-54`), and591/31 Free-form (`p=2.79e-135`). The image-clustered paired95% CI for Overall is **[+12.2733,+14.3957]**; after excluding the first32 samples, the effect remains+13.4141 with CI[+12.3604,+14.4930].
- Inference timing:802.9s; TTFT mean0.048430s, weighted TPOT0.019850s/token, request mean0.110921s.
- Conclusion: a constant domain-level residual is not merely insufficient; it is substantially worse than removing the residual entirely. Dynamic residual vectors are not interchangeable static offsets, and averaging them destroys information needed by both binary and Free-form predictions.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827/eval_interventions/mean-residual`; log `eval_intervention_mean-residual.log`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827__intervention_lagged_memory32

- Commit/config: commit `26ef735`; inference-only intervention preserving the trained CA but replacing each sample's visual and question Memory with Memory from32 samples earlier. Dataset-order audit confirms zero same-image donor pairs after the first32 samples.
- Dataset and split: PathVQA test6,719 with858 image clusters; source training seed/data seed44/42.
- Controlled change: current-sample Memory alignment only; the first32 samples remain normal while6,687 use mismatched Memory. Main VLM image/question inputs remain untouched, so the intervention changes only the Dynamic Prompt branch.
- Overall: **48.5787**; image-clustered95% CI **[47.4625,49.6968]**.
- Yes/No / Free-form: **84.1761 / 12.9282**.
- Yes/No class audit: `yes`84.0859% and `no`84.2820%.
- Per question type: how7.1942, other16.6667, what8.6830, when0.0000, where42.4594, why0.0000, yes/no84.1761; Free-form macro12.5005.
- Intervention audit:6,719 samples seen,6,687 changed,32 warmup samples; no same-image pair among changed samples.
- Paired normal-minus-lagged effect: **+7.9625 Overall**, +5.6514 Yes/No, and **+10.2770 Free-form**. Normal-only/lagged-only correct counts are705/170 Overall (`McNemar p=4.56e-78`),284/94 Yes/No (`p=2.73e-23`), and421/76 Free-form (`p=6.25e-59`). The image-clustered paired95% CI for Overall is **[+6.9923,+8.8855]**. Excluding the first32 samples gives+8.0006 Overall with CI[+7.0470,+8.9920].
- Relative to zero residual: lagged Memory is-0.7144 Overall with clustered CI[-1.5219,+0.0754], trading-3.0042 Yes/No (`p=1.63e-7`) for+1.5788 Free-form (`p=0.00375`). Mismatched domain Memory therefore retains limited generic open-answer signal but loses the much larger benefit of correct sample alignment.
- Inference timing:748.4s; TTFT mean0.049211s, weighted TPOT0.019497s/token, request mean0.102236s.
- Conclusion: current-sample visual/question Memory is causally necessary. Although attention weights harden into an almost fixed modality allocation, the selected Value content remains strongly sample-dependent; Dynamic Prompt conditioning operates through values and residual direction, not through visibly dynamic attention weights.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827/eval_interventions/lagged-memory`; log `eval_intervention_lagged-memory.log`.

### 2026-08-28 - pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_seed44_20260828

- Commit/config: commit `5cb805f`; the Dynamic Prompt20+CA256 method is retained unchanged and jointly trained with a 1,201,155-parameter Sparse Visual MMRL branch. Eight shared 1,024-dimensional Rep tokens use CA128/4 heads over image-level Mean-Pooled visual states and question-only text states, then enter native visual layers5/11/17 (0-based) through independent insert/strip paths. The actual `tanh(gamma_l)` residual scale starts at0.05 and local Relation0.05 is averaged over the three anchors.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change: add the Sparse Visual branch to the exact seed44 Dynamic Prompt protocol. Relative to the aborted zero-scale attempt, only the initial visual residual scale changes from0 to0.05; Prompt LR0.3, Dynamic CA LR3e-4, Sparse Visual LR3e-4, architecture, Relation and all data/evaluation settings remain fixed.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **54.8171 / 88.5760 / 21.1551**; epoch2 **51.5897 / 87.4240 / 15.8583**; epoch3 **54.9129 / 88.2560 / 21.6656**. Epoch3 selected by Overall.
- Test Overall: **56.0202**; image-clustered95% bootstrap CI **[54.7117,57.3767]**.
- Yes/No / Free-form: **88.8757 / 23.1159**.
- Per question type: how13.6691, other55.5556, what17.0741, when16.6667, where64.2691, why4.5455, yes/no88.8757; Free-form question-type macro28.6300.
- Trainable parameters: Soft Prompt51,200 + Dynamic Prompt CA2,634,240 + Sparse Visual1,201,155 = **3,886,595**.
- Training:3 epochs,1,845 optimizer steps, runtime7,122.0s, reported train loss11.2830.
- Late diagnostics at step1,840: Sparse Visual grad norm0.1529 versus Dynamic Prompt0.9881; scales are0.04492/0.07520/0.05029 at layers5/11/17. Applied residual ratios are0.000660/0.034100/0.000321, so the branch is active but overwhelmingly dominated by layer11. Attention entropy is nearly zero at all anchors and visual masses settle near0.25/0.28125/0.50. Mean Sparse residual ratio is0.011694. Scaled Relation is only1.13e-7 and therefore does not materially control optimization.
- Versus Dynamic Prompt20+CA256: **-0.5210 Overall, -0.9518 Yes/No, -0.0893 Free-form**, while Free-form question-type macro changes by+6.2285. The macro gain is concentrated in small how/other/when categories and does not establish a primary-metric improvement.
- Evaluation timing: TTFT mean0.05545s; weighted TPOT0.019804s/token;50.496 decode tokens/s; request mean0.10601s.
- Conclusion: the0.05 initialization successfully fixes branch death, but the active Sparse Visual addition does not improve the main method. It perturbs binary calibration, leaves aggregate Free-form essentially unchanged and lowers both validation and test Overall. Do not run seed45 or tune this sparse architecture further under the current one-week budget; retain it as evidence that stronger internal visual adaptation is not complementary to the Dynamic Prompt interface under shared CE.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`.

Correction (2026-08-28): the immediate recommendation to stop all tuning was too strong. Relative to the original Dynamic Prompt at the same epoch, Sparse Visual epoch1 changes Overall/Yes-No/Free-form by **+2.0131/+0.7360/+3.2866**, before collapsing in epoch2 and recovering in epoch3. This is a positive early signal followed by unstable co-adaptation, not evidence that the visual branch is intrinsically useless. One strict diagnostic is therefore permitted: reduce only Sparse Visual LR from3e-4 to3e-5. This can support an update-timescale/CE-competition explanation if it removes the epoch2 collapse, but cannot by itself prove or disprove gradient conflict.

### 2026-08-28 - pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_sparse_lr3e5_seed44_20260828

- Commit/config: commit `89deb94`; exact same Dynamic Prompt20+CA256 and Sparse Visual layers5/11/17 architecture as the3e-4 control, with Soft Prompt LR0.3, Dynamic CA LR3e-4, initial residual scale0.05 and Relation0.05 unchanged. The only controlled change is Sparse Visual LR **3e-4 -> 3e-5**.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **54.5295 / 88.2880 / 20.8679**; epoch2 **55.1366 / 88.1920 / 22.1761**; epoch3 **56.8461 / 88.8640 / 24.9202**. Epoch3 selected. Unlike the3e-4 control's54.8171 -> 51.5897 -> 54.9129 Overall trajectory, the lower-LR run improves monotonically and removes the epoch2 collapse.
- Test Overall: **57.5086**; image-clustered95% bootstrap CI **[56.1883,58.9046]**.
- Yes/No / Free-form: **89.6193 / 25.3500**.
- Yes/No class audit:1,816 `yes` questions at93.7225% and1,546 `no` questions at84.7995%. Relative to the original Dynamic Prompt, `yes` is unchanged and `no` changes by-0.4528 points; the aggregate binary difference is not significant.
- Per question type: how10.0719, other38.8889, what19.7738, when0.0000, where66.5893, why4.5455, yes/no89.6193; Free-form question-type macro23.3116.
- Trainable parameters: Soft Prompt51,200 + Dynamic Prompt CA2,634,240 + Sparse Visual1,201,155 = **3,886,595**.
- Training:3 epochs,1,845 optimizer steps, runtime7,122.0s, reported train loss11.3922.
- Final diagnostics at step1,840: Soft Prompt norm915.972; Dynamic/Sparse grad norms0.99976/0.01116. Sparse scales remain close to initialization at0.04980/0.05225/0.05151 for layers5/11/17. Applied residual ratios are only0.000464/0.000710/0.001016, with mean0.000730; unlike the3e-4 control, layer11 no longer dominates. Sparse attention entropy remains0.416/0.574/0.565 and visual mass0.306/0.318/0.844. Dynamic attention remains hard with entropy3.98e-5 and visual mass0.44375. Scaled Relation is only1.41e-8 and cannot explain the gain.
- Paired versus the original Dynamic Prompt20+CA256: **+0.9674 Overall, -0.2082 Yes/No, +2.1448 Free-form**. New-only/old-only correct counts are338/273 Overall (`McNemar p=0.00956`),128/135 Yes/No (`p=0.7115`), and210/138 Free-form (`p=0.000134`). Image-clustered paired95% CIs are **[+0.1985,+1.7252] Overall**, [-1.1138,+0.7405] Yes/No, and **[+0.8408,+3.4390] Free-form**. Predictions are textually identical on4,661/6,719 questions (69.37%).
- Paired versus the exact3e-4 Sparse Visual control: **+1.4883 Overall, +0.7436 Yes/No, +2.2341 Free-form**. New-only/old-only counts are397/297 Overall and237/162 Free-form; clustered paired95% CIs are **[+0.6213,+2.3085] Overall** and **[+0.8736,+3.6055] Free-form**.
- Conclusion: the pre-registered LR diagnostic succeeds. Sparse LR3e-4 over-adapts and destabilizes the jointly trained LLM-entry Prompt; reducing only its LR by10x removes the epoch2 collapse and produces statistically positive paired Overall and Free-form gains. This supports an update-timescale mismatch / unstable shared-CE co-adaptation explanation, not the stronger claim that LR alone proves gradient conflict. Because the final Sparse residual averages only0.073%, the result does not yet distinguish a causally useful inference-time visual correction from a training-time regularization/trajectory effect. A same-checkpoint Sparse-off intervention is required before claiming direct visual complementarity.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_sparse_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; test log `eval_test_epoch_3.log`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

Storage correction (2026-08-28): by explicit user direction, the `checkpoints/` and `final/` weight directories for both completed dual-path Sparse Visual runs (LR3e-4 and LR3e-5) were deleted after their results were fully recorded. Evaluation logs, summaries, predictions, comparisons and diagnostics remain at the recorded output roots. The57.5086 result remains valid historical evidence, but its checkpoint can no longer support the planned Sparse-off intervention or direct reuse.

### 2026-08-29 - pathvqa_dynamic_prompt_sparse_visual_single_pass_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828

- Commit/config: commit `e3715c0`; corrected single-pass Sparse Visual baseline using `Strip(Block([Rep; h]))` at visual layers5/11/17. It retains the independently trainable20-token Prompt, Dynamic CA256/8 heads over Mean-Pooled visual/question Memory, shared8x1024 visual Rep base, shared Visual CA128/4 heads and Sparse LR3e-5. Residual Scale, Relation and a direct shared-S text bridge are absent.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change: establish the corrected single-pass reference after removing the historical dual execution path; all three experiments in this suite use the same data order,3 epochs, Prompt LR0.3, Dynamic CA LR3e-4 and Sparse LR3e-5.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **55.0567 / 87.9360 / 22.2719**; epoch2 **56.2071 / 88.0640 / 24.4416**; epoch3 **57.5172 / 89.7280 / 25.3989**. Epoch3 selected.
- Test Overall: **57.7913**; image-clustered95% bootstrap CI **[56.4283,59.1711]**.
- Yes/No / Free-form: **89.7680 / 25.7671**.
- Per question type: how9.3525, other38.8889, what20.0292, when0.0000, where68.4455, why4.5455, yes/no89.7680; Free-form question-type macro23.5436.
- Trainable parameters: Prompt51,200 + Dynamic CA2,634,240 + Sparse Visual1,201,152 = **3,886,592**.
- Training:3 epochs,1,845 optimizer steps, runtime6,914.1s, reported train loss11.3922.
- Diagnostics:93 rows through step1,840. Over the last30 rows, Prompt norm875.14 and grad norm0.01340; Dynamic/Sparse grad norms0.7821/0.5845. Dynamic attention again hardens to normalized entropy5.23e-5 with44.375% visual mass. Dynamic delta/base averages0.4781. Sparse Rep/input and output/input norm ratios average0.7834 and1.1062, confirming a strongly active single-pass branch rather than the historical0.073% residual-scale regime.
- Suite comparison: this is the winner. It exceeds the shared-S-Memory keep-P variant by **+1.6520 Overall**, +0.7436 Yes/No and **+2.5618 Free-form**. Baseline-only/bridge-only correct counts are381/270 Overall and232/146 Free-form; McNemar p-values are1.55e-5 and1.14e-5. The image-clustered paired95% CI for Overall is **[+0.9157,+2.4331]**.
- Historical comparison: its point estimate is+0.2827 Overall and+0.4171 Free-form above the deleted-checkpoint dual-path57.5086 run, but that is an architecture change rather than a same-checkpoint intervention and is not used as causal evidence.
- Evaluation timing: TTFT mean0.079574s, weighted TPOT0.021946s/token,45.566 decode tokens/s, request mean0.137107s.
- Conclusion: the corrected single-pass Sparse Visual method not only preserves the low-LR gain but establishes the new PathVQA seed44 best point estimate. The expensive historical dual block execution, explicit residual scale and Relation are not required for this result. This checkpoint is the supported reference for any additional seed or causal Sparse-off analysis.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_sparse_visual_single_pass_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_shared_s_memory_main_merger_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828

- Commit/config: commit `e3715c0`; exact single-pass baseline plus a direct shared-S text bridge. The8 last-anchor Rep outputs are mapped by the frozen input-aligned main `visual.merger`, reduced to one sample-level S-Memory token and appended as the third Text CA Memory. The independently trainable20-token Prompt remains Query and residual base.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the winning baseline: enable only the parameter-free S-Memory bridge; all trainable tensors, learning rates, initialization, data order and evaluation settings remain identical.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **52.1649 / 86.7200 / 17.7090**; epoch2 **54.3378 / 88.0320 / 20.7403**; epoch3 **56.3189 / 89.3440 / 23.3886**. Epoch3 selected.
- Test Overall: **56.1393**; image-clustered95% bootstrap CI **[54.8226,57.5498]**.
- Yes/No / Free-form: **89.0244 / 23.2052**.
- Per question type: how11.5108, other27.7778, what17.3294, when0.0000, where65.4292, why4.5455, yes/no89.0244; Free-form question-type macro21.0988.
- Trainable parameters: unchanged at **3,886,592**; the reused main Visual Merger remains frozen and is held by weak reference rather than registered again.
- Training:3 epochs,1,845 optimizer steps, runtime6,939.8s, reported train loss11.3500.
- Diagnostics:93 rows through step1,840. The bridge is not dead: last30 mean shared-S parameter grad is0.02356, S-Memory interface grad is0.00905 and total Sparse grad is0.6260. Text CA assigns a stable23.752% attention mass to S-Memory,43.166% to ordinary visual Memory and33.083% to text; normalized entropy is1.91e-4. S-Memory norm averages54.38, while Dynamic delta/base remains a normal0.4417.
- Paired effect versus single-pass baseline: **-1.6520 Overall**, -0.7436 Yes/No and **-2.5618 Free-form**. Bridge-only/baseline-only correct counts are270/381 Overall and146/232 Free-form. The image-clustered paired95% CI for bridge-minus-baseline Overall is **[-2.4331,-0.9157]**; predictions are identical on4,589/6,719 questions.
- Evaluation timing: TTFT mean0.079963s, weighted TPOT0.020459s/token,48.878 decode tokens/s, request mean0.136093s.
- Conclusion: reject the direct shared-S bridge. It receives healthy CE gradients and substantial stable attention yet degrades every primary aggregate metric throughout training. The failure is therefore not branch death; the extra shortcut supplies a competing representation that displaces the already effective native visual/question Memories and worsens joint optimization.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_shared_s_memory_main_merger_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_shared_s_memory_main_merger_no_trainable_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828

- Commit/config: commit `e3715c0`; identical to the keep-P shared-S bridge except the same seed44-initialized20-token P0 is frozen. P0 retains Query positions and prefix length, while Dynamic CA and Sparse Visual remain trainable.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the keep-P bridge: remove only the51,200 trainable Prompt parameters; no sequence length, Query initialization, Memory, backbone or scheduling change.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **51.4140 / 86.3360 / 16.5922**; epoch2 **53.7945 / 87.0400 / 20.6445**; epoch3 **53.8105 / 87.0080 / 20.7084**. Epoch3 narrowly selected; learning essentially plateaus after epoch2.
- Test Overall: **54.7105**; image-clustered95% bootstrap CI **[53.3957,56.0586]**.
- Yes/No / Free-form: **87.4479 / 21.9243**.
- Per question type: how12.2302, other16.6667, what17.3294, when0.0000, where55.9165, why0.0000, yes/no87.4479; Free-form question-type macro17.0238.
- Trainable parameters: Dynamic CA2,634,240 + Sparse Visual1,201,152 = **3,835,392**; Prompt trainable parameters are exactly0.
- Training:3 epochs,1,845 optimizer steps, runtime6,929.2s, reported train loss12.3435.
- Diagnostics: P0 norm remains4.91334 with exactly zero gradient. The bridge remains active: last30 shared-S parameter/interface grad norms are0.01944/0.01780 and S-Memory receives35.000% attention. However Dynamic delta/base explodes to a last30 mean **76.77x** because the frozen token-embedding P0 is tiny relative to the learned residual. Dynamic CA is forced to reconstruct nearly the entire domain Prompt instead of refining a learned base.
- Paired effect versus keep-P bridge: **-1.4288 Overall**, -1.5764 Yes/No and-1.2809 Free-form. No-P-only/keep-P-only counts are297/393 Overall,141/194 Yes/No and156/199 Free-form; McNemar p-values are0.000292,0.00443 and0.0257. The image-clustered paired95% CI for no-P-minus-keep-P Overall is **[-2.2509,-0.6522]**.
- Paired effect versus winning baseline: **-3.0808 Overall**, -2.3200 Yes/No and-3.8427 Free-form; clustered paired95% CI **[-4.0072,-2.2300]**.
- Evaluation timing: TTFT mean0.081610s, weighted TPOT0.019485s/token,51.321 decode tokens/s, request mean0.137366s.
- Conclusion: reject replacing the independent trainable Prompt with frozen P0 plus shared S. The learned P is not redundant scaffolding; it supplies the high-capacity domain anchor around which sample-conditioned CA can operate as a residual. Freezing it harms both binary calibration and open answers and produces a pathological residual/base scale ratio.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_shared_s_memory_main_merger_no_trainable_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_raw_shared_s_separate_residual_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829

- Commit/config: commit `26f95ed`; corrected raw Shared-S keep-P experiment. It preserves the57.79 single-pass baseline's independently trainable P20, original CA256 over Mean-Pooled visual/question Memory, and Sparse Visual injections at layers5/11/17. The unconditioned8x1024 shared parameter bank S is separately mapped by the frozen main `visual.merger` into2 LLM-space tokens; a new zero-output-initialized CA128/4-head branch reads those tokens into20 extra Prompt residuals. No conditioned visual Rep is exported.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the57.79 baseline: add only the raw-S-to-text CA128 residual branch. P20, original visual/question Dynamic CA, Sparse Visual architecture, initialization, data order,3 epochs, Prompt LR0.3, Dynamic/Shared-S CA LR3e-4, Sparse LR3e-5 and evaluator remain matched.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **54.53 / 88.16 / 21.00**; epoch2 **53.49 / 88.58 / 18.51**; epoch3 **54.75 / 88.61 / 21.00**. Epoch3 selected.
- Test Overall: **54.9487**; image-clustered95% bootstrap CI **[53.65,56.26]**.
- Yes/No / Free-form: **88.3105 / 21.5371**; Free-form question-type macro20.6440.
- Per question type: how12.2302, other27.7778, what15.5053, when0.0000, where63.8051, why4.5455, yes/no88.3105.
- Trainable parameters: P51,200 + original Dynamic CA2,634,240 + raw-S CA1,323,520 + Sparse Visual1,201,152 = **5,210,112**.
- Training:3 epochs, runtime6,960s, reported train loss11.44.
- Late diagnostics over the last30 rows: P norm829.44; original Dynamic delta/base0.4598; raw-S residual/base0.03949; raw-S CA grad0.12697; shared-S parameter grad0.03170; mapped-S interface grad0.001309; Sparse total grad0.40464. The branch is active but small. More importantly, visual attention mass at layers5/11/17 becomes47.86%/96.50%/91.16%, versus the baseline's approximately67%/50%/62% final pattern, showing that CE coupling through raw S drives the late visual anchors toward near visual-only routing.
- Paired effect versus the exact57.7913 baseline: **-2.8427 Overall**, -1.4575 Yes/No and **-4.2300 Free-form**. Variant-only/baseline-only correct counts are235/426 Overall,97/146 Yes/No and138/280 Free-form; McNemar p-values are9.94e-14,0.00201 and3.32e-12. The image-clustered paired95% CI for Overall is **[-3.6682,-2.0402]**.
- Evaluation timing: TTFT mean0.053676s, weighted TPOT0.019891s/token,50.273 decode tokens/s, request mean0.106494s.
- Conclusion: reject the separate raw-S residual architecture. This is not branch death, excessive residual magnitude or a mere test fluctuation. The controlled branch addition significantly harms both answer categories and coincides with a major reorganization of visual CA at layers11/17. This is strong evidence for cross-branch optimization interference, but the current run changes both the forward residual and the backward path; a text-side stop-gradient control would be required to distinguish harmful shared gradients from harmful raw-S residual content.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_raw_shared_s_separate_residual_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829

- Commit/config: commit `26f95ed`; corrected pure raw Shared-S experiment. Independent P and frozen P0 are both absent. The unconditioned8x1024 S bank is mapped by the frozen main `visual.merger` into its native2x2560 outputs, which directly form the LLM prefix base and the original CA256 queries over Mean-Pooled visual/question Memory. The same raw S remains the visual Rep basis for layers5/11/17; no conditioned Rep is exported and no token expander is added.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the57.79 baseline: replace independent P20 with the2 native `visual.merger(raw S)` tokens; all visual architecture, Dynamic CA256, Sparse LR3e-5, Dynamic LR3e-4, data order, training length and evaluator remain matched.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **49.93 / 83.78 / 16.18**; epoch2 **52.12 / 88.06 / 16.27**; epoch3 **51.80 / 87.84 / 15.86**. Epoch2 selected.
- Test Overall: **51.6446**; image-clustered95% bootstrap CI **[50.32,53.02]**.
- Yes/No / Free-form: **87.3290 / 15.9071**; Free-form question-type macro12.5477.
- Per question type: how5.7554, other11.1111, what11.7840, when0.0000, where46.6357, why0.0000, yes/no87.3290.
- Trainable parameters: original Dynamic CA2,634,240 + Sparse Visual1,201,152 = **3,835,392**; independent Prompt parameters are exactly0 and effective prefix length is2.
- Training:3 epochs, runtime6,812s, reported train loss13.16.
- Late diagnostics over the last30 rows: mapped raw-S Prompt norm70.33; Dynamic delta/base **2.8340**; shared-S parameter grad0.67544; mapped-S interface grad0.05325; Sparse total grad0.69522. Dynamic attention hardens to81.25% visual mass at the final step. The high gradients prove raw S receives CE, but its3e-5 optimizer group and2-token capacity leave a nearly static, low-capacity base while CA256 must generate a residual almost3x larger than that base.
- Paired effect versus the exact57.7913 baseline: **-6.1467 Overall**, -2.4390 Yes/No and **-9.8600 Free-form**. Variant-only/baseline-only correct counts are268/681 Overall,158/240 Yes/No and110/441 Free-form; McNemar p-values are4.08e-42,4.63e-5 and6.45e-48. The image-clustered paired95% CI for Overall is **[-7.0982,-5.2028]**.
- Evaluation timing: TTFT mean0.053031s, weighted TPOT0.019733s/token,50.675 decode tokens/s, request mean0.102321s.
- Conclusion: reject direct replacement of P20 by native2-token raw S. This run diagnoses a different failure from the keep-P version: the frozen Merger output is trainable through S, but a shared parameter constrained to Sparse LR3e-5 cannot simultaneously behave like the successful LR0.3 textual domain anchor, and two prefix positions are insufficient for PathVQA Free-form adaptation. Raising S to Prompt LR would destroy the controlled visual optimization regime rather than cleanly solve the conflict.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829`; selected checkpoint `checkpoints/epoch_2`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_asymmetric_shared_s_text_owned_visual_readonly_layers5_11_17_slots8_adapter128_seed44_20260829

- Commit/config: commit `8da2390`; the20x2560 trainable LLM Prompt is the sole shared S and remains in the Prompt LR0.3 group. The visual branch can only read `detach(LayerNorm(S))`, maps20 Prompt tokens to8 slots with a learned Token Mixer and2560->128->1024 adapter, and then uses the existing CA128/4-head single-pass injections at visual layers5/11/17. Dynamic Text CA256 remains at3e-4, while the visual adapter, Visual CA and layer embeddings use3e-5. There is no independent visual S, second P, Relation or `visual.merger`.
- Dataset and split: PathVQA train19,654; fixed full validation6,259 with832 image clusters. Per the current development protocol, only epoch3 Validation was evaluated; Test was not run.
- Seed / data seed:44 /42.
- Controlled change versus the single-pass baseline: replace the independent8x1024 visual Rep bank with a detached low-rank projection of the existing P20/S, while preserving Prompt length/init/LR, Dynamic CA, visual anchor layers, Visual CA, data order and3-epoch schedule.
- Fixed epoch3 Validation Overall: **56.8142**; image-clustered standalone95% CI **[55.3244,58.2287]**.
- Yes/No / Free-form: **88.2240 /25.4946**; Free-form question-type macro19.6369.
- Binary class audit: among1,712 `yes` references, baseline/new accuracies are95.3271/95.6776 (+0.3505); among1,413 `no` references they are82.9441/79.1932 (**-3.7509**). The aggregate binary loss is therefore an amplified affirmative bias, not a uniform recognition decline.
- Per question type: how9.3023, other14.2857, what19.5447, when0.0000, where69.9267, why4.7619, yes/no88.2240.
- Trainable parameters: S/P51,200 + Dynamic CA2,634,240 + Sparse Visual1,651,872 = **4,337,312**.
- Training:3 epochs,1,845 optimizer steps, runtime6,893.1s, reported train loss11.4963.
- Late diagnostics over the last30 rows: S/P grad0.01487; Dynamic/Sparse/visual-adapter grad norms0.7926/0.5937/0.2124; the visual-to-S gradient-block audit is1.0 for every diagnostic row. Token-Mixer normalized entropy0.4435 and adapter Rep norm3.399 confirm a live, non-collapsed adapter. Dynamic delta/base0.4689, visual mass46.876% and entropy0.000197 remain close to the baseline text-side branch.
- Training trajectory: visual text-dominance exists from the first0-280-step window, where layers5/11/17 visual mass is35.58%/29.13%/29.35% versus baseline59.48%/54.76%/62.49%; it is therefore an architectural/query-source bias rather than a late-training accident. Adapter Rep norm rises0.623 ->3.366 and its gradient0.0427 ->0.1974, while Token-Mixer entropy remains effectively fixed0.4443 ->0.4435. The feature projections amplify the branch, but the arbitrary20-to-8 token routing does not discover meaningful specialization.
- Visual-side mechanism shift: layers5/11/17 Visual CA mass becomes28.75%/27.31%/28.65%, versus baseline68.43%/50.26%/61.09%. Rep/input ratios increase to1.588/1.486/0.678 versus baseline1.173/0.831/0.346. The detached text-owned anchor therefore makes the visual branch stronger but markedly text-memory dominated.
- Paired effect versus the exact single-pass baseline epoch3 Validation57.5172: **-0.7030 Overall**, -1.5040 Yes/No and +0.0957 Free-form. New-only/baseline-only correct counts are229/273 Overall,82/129 Yes/No and147/144 Free-form; McNemar p-values are0.0549,0.00148 and0.9067. The image-clustered paired95% CI for Overall is **[-1.4045,+0.0161]**.
- Evaluation timing: TTFT mean0.053565s, weighted TPOT0.019811s/token,50.478 decode tokens/s, request mean0.105524s.
- Conclusion: gradient decoupling works as designed and rescues raw hard sharing from a6.15-point failure to a near-baseline result, but it does not improve the supported method. Free-form is statistically tied while binary calibration degrades significantly. The remaining failure is forward over-coupling, not backward gradient pollution: replacing the independent visual Rep basis with text-derived S causes Visual CA to abandon visual Memory and amplifies Rep strength. Do not add seed45 for this pure replacement architecture. Any final rescue must preserve the independent visual basis and add only a bounded detached-S residual; that would be a new hypothesis rather than a retune of this run.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_asymmetric_shared_s_text_owned_visual_readonly_layers5_11_17_slots8_adapter128_seed44_20260829`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_validation/epoch_3`.

### 2026-08-30 - pathvqa_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260829

- Commit/config: commit `deebda4`; preserve the exact supported private baseline components (P20 at LR0.3, Text CA256/8 heads at3e-4, independent visual S8, shared private Visual CA128/4 heads at3e-5 and single-pass insert/strip at visual layers5/11/17). Add a32x1024 sample-conditioned workspace Z that is sequentially updated at the three anchors by independent full-token Cross-Attention, Self-Attention and FFN4096 blocks. Full current-layer visual tokens and full question-token embeddings are the workspace Memory. Separate Text CA1024/16-head and three Visual CA1024/16-head residual readers consume Z; their output projections start at exact zero, and Z is never concatenated into the original private CA Memories.
- Dataset and split: PathVQA train19,654; fixed full validation6,259 with832 image clusters. Only epoch3 Validation was evaluated under the current protocol; Test was not run.
- Seed / data seed:44 /42.
- Controlled change versus the single-pass baseline: add only the full-capacity shared workspace trunk and its four separate residual readers. The private P, private visual S, original Text/Visual CAs, anchor layers, injection rule, data order and their learning rates remain present. Workspace LR is1e-4.
- Fixed epoch3 Validation Overall: **59.4504**; image-clustered standalone95% CI **[58.0231,60.8857]**.
- Yes/No / Free-form: **90.6240 /28.3663**; Free-form question-type macro19.4184.
- Per question type: how8.5271, other14.2857, what23.0377, when0.0000, where70.6601, why0.0000, yes/no90.6240.
- Trainable parameters: Prompt51,200 + Dynamic CA2,634,240 + private Sparse Visual1,201,152 + Shared Workspace73,009,664 = **76,896,256**.
- Training:3 epochs,1,845 optimizer steps, runtime7,322s, reported train loss10.93. This is408s /5.9% slower than the exact baseline's6,914.1s despite19.8x trainable parameters.
- Paired effect versus the exact single-pass baseline epoch3 Validation57.5172/89.7280/25.3989: **+1.9332 Overall, +0.8960 Yes/No and +2.9675 Free-form**. Workspace-only/baseline-only correct counts are352/231 Overall,126/98 Yes/No and226/133 Free-form; exact McNemar p-values are6.13e-7,0.0710 and1.05e-6. Image-clustered paired95% CIs are **[+1.1747,+2.7132] Overall**, [0.0000,+1.7948] Yes/No and **[+1.8602,+4.1721] Free-form**. Predictions are textually identical on4,478/6,259 questions (71.55%).
- Binary class audit: `yes` improves by+1.9276 points (45 new-only /12 baseline-only, p=1.31e-5, paired clustered CI[+1.0782,+2.8324]); `no` changes by-0.3539 (81/86, p=0.7570, CI[-2.1161,+1.5626]). The aggregate binary gain is affirmative-side, but the primary Overall improvement is not a binary-calibration artifact because the larger Free-form gain is independently significant.
- Mechanism trajectory: over the first30 versus last30 diagnostic rows, Workspace grad remains healthy0.6160->0.5745 while private Sparse Visual grad falls0.1244->0.0615. Workspace visual delta/private-Rep rises2.836x->5.246x and total Rep/input rises2.479x->5.472x, yet frozen visual Block output/input stays stable1.112x->1.107x. The shared visual path therefore becomes functionally dominant without causing a numerical backbone-output explosion.
- Workspace behavior: full-token attention becomes strongly selective (layer5/11/17 normalized entropy first30 0.888/0.906/0.821 -> last30 0.377/0.365/0.226), while final text and visual readers remain nearly uniform over32 Z slots (about0.9998 normalized entropy). Current evidence supports a useful deep full-token workspace with global/mean-like downstream readout; it does not yet establish semantic specialization among the32 slots.
- Final-step diagnostics: Dynamic and Workspace-text delta/base are0.3477 and0.2219; private Sparse grad0.0438 versus Workspace0.5790; layer5/11/17 Workspace visual delta/private-Rep are7.310/5.305/2.671; aggregate Rep/input4.977 and Block output/input1.085. The high score coexists with shared-path takeover, so retaining private branches is structurally true but cooperative contribution from both paths remains unproven.
- Evaluation timing: TTFT mean0.058198s, weighted TPOT0.019724s/token,50.698 decode tokens/s, request mean0.104538s. Timing is recorded but not used for a strict cross-run claim because the baseline was evaluated under a different runtime session.
- Conclusion: this is the new PathVQA seed44 development best and validates the high-capacity-first strategy. It clears the pre-registered success rule with a significant Overall gain driven primarily by Free-form improvements. Before parameter reduction, run seed45 for replication and perform same-checkpoint inference interventions that separately disable/scale the Workspace text and visual residual readers; these are required to distinguish useful shared coordination from a high-capacity visual takeover and to choose a principled slimming target.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260829`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_validation/epoch_3`.

### 2026-08-30 - pathvqa_lora_full_model_attention_r8_seed44_20260830

- Dataset and split: PathVQA train19,654; fixed full validation6,259 with832 image clusters. Only epoch3 Validation was evaluated under the current protocol; Test was not run.
- Seed / data seed:44 /42.
- Controlled method: full-model Attention LoRA-r8 with alpha16, dropout0.05 and LR1e-4. It targets qkv/proj in all24 visual Attention layers and q/k/v/o in all36 LLM Self-Attention layers, for48 visual +144 language =192 selected Linear modules. Micro-batch1 with gradient accumulation32 gives effective batch32; training lasts3 epochs.
- Fixed epoch3 Validation Overall: **59.3386**; image-clustered standalone95% CI **[57.8839,60.7231]**.
- Yes/No / Free-form: **92.1920 /26.5795**; Free-form question-type macro23.5619.
- Per question type: how12.4031, other21.4286, what19.7802, when7.6923, where75.3056, why4.7619, yes/no92.1920.
- Trainable parameters: expected and audited actual **7,077,888**, or0.159236% of4,444,893,696 total parameters.
- Training:3 epochs,1,845 optimizer steps, runtime14,827.47s, reported train loss0.588059.
- Paired effect of Full Workspace minus LoRA on the identical Validation questions: **+0.1118 Overall**, **-1.5680 Yes/No** and **+1.7869 Free-form**. Workspace-only/LoRA-only correct counts are329/322 Overall,103/152 Yes/No and226/170 Free-form; exact McNemar p-values are0.8141,0.00258 and0.00564. Image-clustered paired95% CIs are **[-0.7669,+0.9717] Overall**, **[-2.5633,-0.5853] Yes/No** and **[+0.2844,+3.2611] Free-form**. Their predictions are textually identical on4,137/6,259 questions (66.10%).
- Capability decomposition: Full Workspace exceeds LoRA by+3.2575 on `what` but trails by-4.6455 on `where`. Within Yes/No, Workspace is+2.9206 on `yes` references and-7.0064 on `no` references, so LoRA's aggregate binary advantage mainly reflects markedly better negative-class calibration rather than a uniform gain over both labels.
- Evaluation timing: TTFT mean0.057489s, weighted TPOT0.030869s/token,32.395 decode tokens/s, request mean0.131781s. Timing is descriptive only because the two methods were evaluated in different runtime sessions.
- Conclusion: Full Workspace and full-model LoRA-r8 are statistically tied on Overall at seed44, but not functionally equivalent. LoRA is10.87x smaller and significantly stronger on Yes/No, while Full Workspace is significantly stronger on Free-form and `what`, which is the more relevant evidence for sample-conditioned multimodal coordination. This prevents an Overall-only superiority claim but establishes LoRA as a strong, genuinely matched performance baseline rather than evidence that the Workspace idea failed. Cross-dataset SLAKE validation now has higher priority than PathVQA seed45.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r8_seed44_20260830`; checkpoint `checkpoints/epoch_3`; training log `train.log`; evaluation `eval_validation/epoch_3` and `eval_validation_epoch_3.log`.

### 2026-08-30 - slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260830

- Commit/config: commit `4a68586`; exact cross-dataset transfer of the PathVQA Full Workspace architecture. It keeps P20, private Text CA256/8 heads, independent visual S8, private Visual CA128/4 heads and single-pass visual injections at layers5/11/17. The added Workspace uses Z32x1024, three independent full-token Cross-Attention/Self-Attention/FFN4096 blocks, one Text reader CA1024/16 heads and three Visual reader CAs1024/16 heads. All architecture and optimizer hyperparameters were fixed before seeing SLAKE results.
- Dataset and split: SLAKE train.json for3 training epochs; official test.json with2,094 questions for the single fixed epoch3 evaluation. Test contains836 CLOSED /1,258 OPEN,267 KVQA /1,827 VQA,1,061 EN /1,033 ZH samples.
- Seed / data seed:44 /42.
- Controlled change: transfer the complete PathVQA Workspace method to SLAKE without per-dataset architecture or learning-rate tuning. Prompt/Dynamic/private-Sparse/Workspace learning rates remain0.3/3e-4/3e-5/1e-4; effective batch is32.
- Official Test Overall: **78.37** (1,641/2,094 correct).
- CLOSED / OPEN: **84.09 /74.56**.
- KVQA / VQA: **63.30 /80.57**.
- EN / ZH: **78.79 /77.93**.
- Trainable parameters: **76,896,256**. Exact decomposition is P20 51,200; private Text CA2,634,240; private Sparse Visual1,201,152; three Workspace update blocks50,396,160; three Workspace Visual readers12,598,272; Workspace Text reader7,349,760; text-to-Workspace projection/norm2,627,584; Workspace seeds/type/layer embeddings37,888.
- Training:3 epochs,924 optimizer steps, runtime5,195s, reported train loss3.335.
- Paired effect versus SLAKE Static Prompt Tuning seed44 74.40: **+3.9637 Overall** with145 Workspace-only and62 Static-only correct questions, exact McNemar `p=7.65e-9`. Effects are +3.4689 CLOSED (`p=0.00169`), +4.2925 OPEN (`p=1.64e-6`), +0.7491 KVQA (`p=0.856`), +4.4335 VQA (`p=9.30e-10`), +3.3930 EN (`p=0.000223`) and +4.5499 ZH (`p=1.38e-5`). The gain is robust but is concentrated in visual VQA rather than knowledge questions.
- Mechanism diagnostics over47 rows, steps0..920: Workspace grad remains active but declines0.6864 ->0.3625, while private Sparse grad rises0.1711 ->0.2005. Dynamic Prompt attention hardens and remains text-heavy in aggregate (last15 visual/text mass30.63%/69.38%). In contrast, Workspace full-token updates become almost visual-only: last15 visual mass at layers5/11/17 is95.22%/99.48%/99.85%. The Workspace Text-reader residual/base falls22.49% ->9.00%, and all Text/Visual readers remain nearly uniform over Z32 (normalized entropy about0.9997-0.9999). Workspace visual residual/private-Rep ends at0.558x/2.080x/1.038x by layer.
- Evaluation timing: TTFT mean0.073514s, weighted TPOT0.019453s/token,51.407 decode tokens/s, request mean0.121174s.
- Conclusion: cross-dataset performance generalization is established: the fixed Workspace significantly improves both PathVQA and SLAKE, especially OPEN/VQA questions. The current76.90M implementation is nevertheless not an acceptable final efficiency point, and the intended multimodal-workspace interpretation remains unsupported on SLAKE because full-token Workspace updates nearly ignore text and downstream slot readers are uniform. Preserve this checkpoint as the high-capacity teacher/reference. Before any width or slot sweep, use same-checkpoint reader interventions to identify indispensable branches; then prioritize recurrent sharing of the three update blocks and sharing of the three Visual readers. Modality-balanced visual/text attention is a separate performance/mechanism correction and must not be conflated with the first parameter-reduction control.
- Strategy correction (2026-08-30): parameter reduction is intentionally postponed. With approximately475 visual versus15 question tokens per batch sample, a joint softmax assigns about96.9% aggregate mass to vision even under equal logits, closely matching the observed95-99.8% visual mass. The next controlled experiment therefore keeps full capacity and replaces joint visual/text competition with separate modality softmax branches plus learned fusion. Weight sharing, reader removal and FFN reduction resume only after the highest-performing architecture is fixed.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260830`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_test/epoch_3` and `eval_test_epoch_3.log`.

### 2026-08-30 - slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s8_seed44_20260830

- Commit/config: commit `d2641a3`; retrained layer ablation of the exact SLAKE Full Workspace method. Private visual injection, Workspace update and Workspace visual readout are all retained only at visual layer17; layers5/11 and their independent Workspace blocks/readers are absent. P20, private Text CA256/8 heads, independent visual S8, private Visual CA128/4 heads, Z32x1024, FFN4096, Workspace Text reader and all learning rates remain unchanged.
- Dataset and split: SLAKE train.json for3 epochs; official test.json with2,094 questions evaluated once at epoch3.
- Seed / data seed:44 /42.
- Controlled change: visual anchor layers `[5,11,17] -> [17]`, which also changes Workspace blocks/readers `3 -> 1`; no other architecture, optimizer, data-order or evaluation change.
- Official Test Overall: **78.37** (1,641/2,094 correct), exactly equal to the three-layer reference.
- CLOSED / OPEN: **84.57 /74.24**.
- KVQA / VQA: **62.92 /80.62**.
- EN / ZH: **79.17 /77.54**.
- Trainable parameters: **34,895,872**, down42,000,384 / **54.62%** from76,896,256.
- Training:3 epochs,924 optimizer steps, runtime4,705.84s, reported train loss3.2140. Runtime is489.16s /9.42% lower than the three-layer run's5,195s.
- Paired effect versus the exact three-layer Workspace: Overall difference **0.0000**, with70 layer17-only-only and70 three-layer-only correct questions, exact McNemar `p=1.0`. Normalized predictions are identical on1,834/2,094 questions. Subgroup differences are all nonsignificant: CLOSED+0.4785 (`p=0.683`), OPEN-0.3180 (`p=0.747`), KVQA-0.3745 (`p=1.0`), VQA+0.0547 (`p=1.0`), EN+0.3770 (`p=0.712`) and ZH-0.3872 (`p=0.728`).
- Diagnostics:47 rows through step920. First15 -> last15 Workspace grad remains healthy0.750 ->0.711. The single layer17 Workspace update becomes less visual-dominated over training but remains biased: visual/text mass97.02%/2.98% ->93.64%/6.36%, while normalized update entropy falls0.884 ->0.693. Workspace norm grows80.46 ->151.02 rather than reaching the three-layer final depth's approximately420. The layer17 Workspace visual residual/private-Rep ratio rises0.537 ->2.434. Its Z32 visual-reader entropy remains nearly uniform0.99979 ->0.99971. Crucially, Workspace Text residual/base rises0.194 ->0.293 instead of decaying to the three-layer run's approximately0.09.
- Conclusion: layers5/11 are **removable under retraining** at seed44, but this experiment does not establish that they are inactive inside the trained three-layer model. It jointly removes their Workspace updates and direct visual writes, while allowing layer17 to compensate: the last-window layer17 Workspace residual/private-Rep ratio rises from1.038 in the three-layer model to2.434 in layer17-only; the original layer11 ratio2.080 also proves that branch was not dead. The result therefore establishes functional redundancy/non-identifiability, not the causal reason for it. Candidate mechanisms are sequential overwrite of a single Z bank, nearly uniform Z32 readout, repeated visual-dominated updates from the same static question embeddings, and attenuation/compensation of early insert-strip writes. Same-checkpoint path interventions are required before rejecting multi-stage coordination.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s8_seed44_20260830`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_test/epoch_3` and `eval_test_epoch_3.log`.

### 2026-08-30 - slake_full_workspace_path_interventions_seed44_20260830

- Commit/config: commit `0b3d5ec`; six inference-only interventions reuse the exact trained three-layer Full Workspace checkpoint and its fixed official SLAKE Test predictions. No weight, input, decoding setting or checkpoint changes. The normal reference is **78.3668** Overall from `slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260830`.
- Dataset and split: official SLAKE test.json, 2,094 questions: 836 CLOSED / 1,258 OPEN, 267 KVQA / 1,827 VQA, 1,061 EN / 1,033 ZH.
- Seed / data seed: 44 / 42; inference interventions only, with no retraining.
- `visual_write_off_l5`: disable only the layer5 Workspace visual residual write after preserving its Z update and all later paths. Overall **78.5578**; CLOSED / OPEN **84.3301 / 74.7218**; KVQA / VQA **63.6704 / 80.7334**; EN / ZH **78.8878 / 78.2188**. Effect versus normal: **+0.1910**, intervention-only/reference-only correct **10/6**, exact McNemar `p=0.4545`.
- `visual_write_off_l11`: disable only the layer11 Workspace visual residual write. Overall **78.3190**; CLOSED / OPEN **84.3301 / 74.3243**; KVQA / VQA **63.6704 / 80.4598**; EN / ZH **78.7936 / 77.8316**. Effect **-0.0478**, exclusive correct **19/20**, `p=1.0`.
- `visual_write_off_l5_l11`: disable both early direct Workspace visual writes while preserving both Z updates and the layer17 write. Overall **78.0802**; CLOSED / OPEN **84.3301 / 73.9269**; KVQA / VQA **63.2959 / 80.2408**; EN / ZH **78.5108 / 77.6379**. Effect **-0.2865**, exclusive correct **16/22**, `p=0.4177`.
- `workspace_update_off_l5`: bypass only the layer5 Workspace update while retaining the layer5 visual reader/write over the incoming Z and all later paths. Overall **78.5578**; CLOSED / OPEN **84.4498 / 74.6423**; KVQA / VQA **64.0449 / 80.6787**; EN / ZH **78.8878 / 78.2188**. Effect **+0.1910**, exclusive correct **7/3**, `p=0.3438`.
- `workspace_update_off_l11`: bypass only the layer11 Workspace update. Overall **78.5100**; CLOSED / OPEN **84.0909 / 74.8013**; KVQA / VQA **64.0449 / 80.6240**; EN / ZH **78.6993 / 78.3156**. Effect **+0.1433**, exclusive correct **8/5**, `p=0.5811`.
- `workspace_update_off_l5_l11`: bypass both early Z updates while preserving the direct visual readers/writes and the complete layer17 Workspace path. Overall **78.4145**; CLOSED / OPEN **84.8086 / 74.1653**; KVQA / VQA **63.2959 / 80.6240**; EN / ZH **78.5108 / 78.3156**. Effect **+0.0478**, exclusive correct **20/19**, `p=1.0`.
- Diagnostic scale intervention: normal-like visual-write interventions retain Workspace text-memory norm near **493**. Bypassing layer11, layer5 or both early updates changes that norm to **328.26 / 349.81 / 180.78**, respectively, while Overall remains statistically unchanged. Workspace Text-reader residual/base is **0.0892** for normal-like runs and **0.0896 / 0.0788 / 0.0711** for the three update-bypass runs. Its Z32 attention remains effectively uniform (`entropy_norm` approximately 1.0).
- Diagnostic limitation: `workspace_debug_means` captured only final Text-reader metrics. The intended per-layer `workspace_transition_*` LayerNorm-cosine and visual-delta cosine metrics are absent from the saved comparison, so this suite cannot directly distinguish repeated information from overwritten or cancelling layer updates. This logging gap does not affect the intervention configurations, predictions or paired scores.
- Conclusion: the layer5 and layer11 **Z updates are causally dispensable in the already-trained three-layer checkpoint**, not merely removable after retraining. They strongly alter Workspace magnitude/history but add no identifiable accuracy, consistent with layer17 overwrite plus scale-insensitive LayerNorm and near-uniform final Z readout. The two early direct visual writes are also individually dispensable; disabling both produces only a small nonsignificant `-0.2865`, leaving weak joint contribution/co-adaptation possible but unproven. Do not claim the modules are numerically dead: claim that their learned updates/writes contain no indispensable task information under these controlled interventions.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_full_workspace_path_interventions_seed44_20260830`; manifest `intervention_manifest.tsv`; per-variant predictions/summaries under the six named subdirectories; aggregate paired results `workspace_path_comparison.tsv` and `workspace_path_comparison.json`.

### 2026-08-30 - slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s20_seed44_20260830

- Commit/config: commit `7b3f00c`; exact layer17-only Full Workspace configuration with the sole controlled change `private visual S / inserted Rep tokens 8 -> 20`. Workspace remains Z32x1024 with one full-token CA/Self-Attention/FFN4096 Block, P20 and both text readers are unchanged, and all learning rates/data order remain matched.
- Dataset and split: SLAKE train.json for3 epochs; official test.json with2,094 questions evaluated once at epoch3.
- Seed / data seed:44 /42.
- Official Test Overall: **78.22** (1,638/2,094 correct).
- CLOSED / OPEN: **82.78 /75.20**.
- KVQA / VQA: **62.92 /80.46**.
- EN / ZH: **79.17 /77.25**.
- Trainable parameters: **34,908,160**, exactly12,288 above S8 because only12 additional1024-dimensional private Rep parameters are introduced.
- Training:3 epochs,924 optimizer steps, runtime4,696s, reported train loss3.193.
- Paired effect versus exact S8+Z32 layer17-only reference: **-0.1433 Overall**, with77 S20-only and80 S8-only correct questions, exact McNemar `p=0.8732`. CLOSED changes84.57 ->82.78 (**-1.7943**), exclusive correct21/36, `p=0.0627`; OPEN changes74.24 ->75.20 (**+0.9539**), exclusive correct56/44, `p=0.2713`. KVQA has identical168 correct with12/12 churn; VQA is-0.1642 with65/68, `p=0.8624`; EN has identical840 correct with31/31 churn; ZH is-0.2904 with46/49, `p=0.8376`.
- Mechanism comparison, first15 -> last15 diagnostic means. S20 private Rep/input grows0.727 ->1.829 versus S8 0.407 ->1.418, but frozen Visual Block output/input remains effectively identical at1.163 versus1.164. Private Visual CA entropy hardens much further0.497 ->0.165 versus S8 0.775 ->0.361, while its visual-memory mass falls31.65% ->27.00% versus S8 43.15% ->42.67%; the extra queries become more selective but more text-dominated. Workspace visual residual/private-Rep grows1.073 ->2.931 versus S8 0.537 ->2.434, while its Z32 read entropy remains uniform at0.9997. Sparse late gradient remains healthy0.0631 versus S8 0.0668, but Workspace late gradient falls0.711 ->0.442 and Workspace Text residual/base falls from the S8 last-window0.2928 to S20 0.1402.
- Evaluation timing: TTFT mean0.067927s, weighted TPOT0.020225s/token,49.444 decode tokens/s, request mean0.117485s. Timing is descriptive because runs were evaluated in separate runtime sessions.
- Conclusion: reject the hypothesis that8 private visual Rep positions are the current expression-capacity bottleneck. Expanding to20 makes the visual path stronger and more selective and changes157 question outcomes, so the branch is neither gradient-dead nor functionally inert. It nevertheless produces no net accuracy and shifts capability from CLOSED toward OPEN while suppressing the Workspace Text contribution. The supported diagnosis is **active but non-complementary visual writing / cross-branch compensation**, not “the visual side does nothing.” Per the pre-registered stop rule, do not run the combined S20+Z8 training experiment. Test final visual and text path necessity with same-checkpoint inference interventions before another retraining change.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s20_seed44_20260830`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_test/epoch_3`; paired comparison `eval_test/workspace_path_comparison.json` and `.tsv`.

### 2026-08-30 - slake_17only_final_path_interventions_seed44_20260830

- Commit/config: commit `c24940c`; four inference-only interventions reuse the exact trained `slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s8_seed44_20260830` checkpoint and its fixed official SLAKE Test predictions. No training, weight, input, decoding or checkpoint change is involved. Disabled paths are still computed for diagnostics and are zeroed only at their final forward write.
- Dataset and split: official SLAKE test.json, 2,094 questions: 836 CLOSED / 1,258 OPEN, 267 KVQA / 1,827 VQA, 1,061 EN / 1,033 ZH.
- Seed / data seed: 44 / 42; inference interventions only.
- Reference: layer17-only S8+Z32 Overall **78.3668**; CLOSED / OPEN **84.57 / 74.24**; KVQA / VQA **62.92 / 80.62**; EN / ZH **79.17 / 77.54**.
- `workspace_visual_off_l17`: preserve the Workspace update, private visual Rep insertion and Workspace Text reader, but zero only the layer17 Workspace visual residual. Overall **78.0802**; CLOSED / OPEN **84.9282 / 73.5294**; KVQA / VQA **62.1723 / 80.4050**; EN / ZH **78.6051 / 77.5411**. Effect versus reference **-0.2865**, intervention-only/reference-only correct **21/27**, exact McNemar `p=0.4709`.
- `all_visual_rep_off_l17`: preserve the Workspace update and Workspace Text reader, but disable both private and Workspace Rep insertion into visual Block17. Overall **77.8415**; CLOSED / OPEN **84.8086 / 73.2114**; KVQA / VQA **61.0487 / 80.2956**; EN / ZH **78.4166 / 77.2507**. Effect **-0.5253**, exclusive correct **14/25**, `p=0.1081`.
- `workspace_text_off`: preserve the complete private and Workspace visual paths, but zero the Workspace Text residual at the LLM prompt interface. Overall **76.3610**; CLOSED / OPEN **83.4928 / 71.6216**; KVQA / VQA **61.4232 / 78.5441**; EN / ZH **78.1338 / 74.5402**. Effect **-2.0057**, exclusive correct **56/98**, exact McNemar `p=0.0008935`.
- `workspace_visual_text_off`: preserve the private visual Rep path and Workspace update, but zero both Workspace visual and Workspace Text residual writes. Overall **76.2178**; CLOSED / OPEN **83.1340 / 71.6216**; KVQA / VQA **61.4232 / 78.3799**; EN / ZH **78.1338 / 74.2498**. Effect **-2.1490**, exclusive correct **56/101**, exact McNemar `p=0.0004104`.
- Diagnostics/causal boundary: disabling all added visual Rep writes is a small nonsignificant change, whereas disabling the Workspace Text write causes a reproducible approximately2-point loss, especially on OPEN and Chinese questions. The useful Workspace output is therefore routed primarily through the LLM prompt interface rather than through direct Rep insertion into visual Block17. This does **not** show that visual information is unnecessary: `all_visual_rep_off_l17` still lets the Workspace update read the current visual tokens, and its resulting Z still conditions the Workspace Text residual; the frozen VLM's ordinary visual tokens also remain unchanged.
- Conclusion: direct visual writing is active but not causally indispensable at this checkpoint, and increasing private Rep capacity cannot repair that lack of complementarity. The current performance core is `multimodal layer17 Workspace update -> Workspace Text residual -> LLM prompt`. Before retraining or shrinking the model, the next decisive same-checkpoint control must remove or sample-mismatch only the visual Memory used by the Workspace update while preserving its Text reader. That separates a genuinely visual-conditioned dynamic prompt from a high-capacity text-conditioned prompt. No additional Rep-count sweep is justified.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_17only_final_path_interventions_seed44_20260830`; manifest `intervention_manifest.tsv`; per-variant predictions and logs under `workspace_visual_off_l17`, `all_visual_rep_off_l17`, `workspace_text_off` and `workspace_visual_text_off`; aggregate paired results `workspace_path_comparison.tsv` and `workspace_path_comparison.json`.

### 2026-08-31 - slake_directional_concat_workspace_z10_d1024_l17_private_p20_s8_seed44_20260830

- Commit/config: commit `1e65bbe`; Directional Concat Workspace uses only visual layer17. Full question embedding tokens are attention-pooled into `Q10x1024`, which performs one standard CA1024/16 over the complete layer17 visual tokens as K/V to form shared `Z10`. A zero-output-initialized `1024->1024` MLP maps Z to ten visual Workspace tokens anchored by `A_v10`, concatenated with private `S8` before one insert/block/strip pass. A separate zero-output-initialized `1024->2560` MLP maps the same Z to ten LLM Workspace tokens anchored by `A_t10`, concatenated after private `P20` to form Prompt30. The old Private Text CA, Private Visual CA, Workspace update Block, Workspace Visual Reader and Workspace Text Reader are absent.
- Dataset and split: SLAKE train.json for3 epochs; official test.json with2,094 questions evaluated once at epoch3.
- Seed / data seed:44 /42.
- Controlled change: replace the `34,895,872`-parameter layer17-only Full Workspace `S8+Z32` architecture with the agreed directional single-CA and explicit token-concat architecture; preserve frozen base VLM, layer17 location, private `P20/S8`, data order, epoch count and learning rates `P/A_t=0.3`, `S8=3e-5`, shared Workspace `1e-4`.
- Official Test Overall: **77.17** (1,616/2,094 correct).
- CLOSED / OPEN: **82.89 /73.37**.
- KVQA / VQA: **61.05 /79.53**.
- EN / ZH: **78.32 /75.99**.
- Trainable parameters: **12,726,784**: Prompt group76,800, private visual S8 8,192, shared Directional Workspace12,641,792. This is22,169,088 / **63.53%** fewer parameters than the34.90M layer17-only reference.
- Training/runtime:3 epochs,924 optimizer steps, runtime4,715.40s, reported train loss3.6238. Runtime is effectively unchanged from the34.90M reference's4,705.84s because full visual-token CA and the frozen backbone dominate; parameter reduction did not produce training acceleration. Evaluation TTFT mean0.066464s, weighted TPOT0.019955s/token and50.113 decode tokens/s.
- Paired effect versus the exact layer17-only Full Workspace78.3668 reference: **-1.1939 Overall**, with77 Directional-only and102 reference-only correct questions, exact McNemar `p=0.07254`. The loss is directionally consistent across every subgroup: CLOSED-1.67, OPEN-0.87, KVQA-1.87, VQA-1.09, EN-0.85 and ZH-1.55. It is a clear one-seed downward trend but does not cross the conventional paired `p<0.05` threshold.
- Paired effect versus Static Prompt Tuning seed44 74.40: **+2.7698 Overall**, with139 Directional-only and81 Static-only correct questions, exact McNemar `p=0.0001118`. The shared multimodal route therefore retains substantial task value and cannot be reduced to the extra ten static prompt positions.
- Diagnostics over47 rows, steps0..920: shared Workspace gradient remains clipped near1.0 from the first15 to last15 windows, while private visual gradient is stable0.01776 ->0.01740. Text pooling entropy decreases0.9196 ->0.8275, but final evaluation still averages0.8630. The visual CA output/query norm ratio rises2.10 ->2.53 in training and is3.35 in evaluation, so Z remains visual-dominated despite text-derived queries. Z slot pairwise cosine stays very high0.970 ->0.943 and averages0.941 at evaluation, indicating weak slot specialization. Text dynamic delta/anchor grows0.330 ->0.531 and averages0.505 at evaluation; visual dynamic delta/anchor grows15.98 ->28.66 and averages28.17. Visual Prompt/input grows0.192 ->0.337 while frozen Visual Block output/input remains unchanged1.164 ->1.163. The implementation is active, but its shared slots are correlated and both output writes become strong rather than bounded residuals.
- Diagnostic limitation: training-time `workspace_visual_attention_entropy_norm` is finite in only14/47 rows because padded probabilities are multiplied to zero before applying `log`, producing diagnostic-only `0*log(0)` NaNs. Evaluation batches produce the finite0.7592 mean. This logging defect does not enter the forward result or loss, but must be fixed before relying on the training entropy trend.
- Conclusion: as the next high-score architecture this run misses the target; the old34.90M layer17-only Workspace remains the SLAKE performance reference. As a compact architecture it is nevertheless a strong Pareto point, preserving a statistically significant+2.77 gain over Static Prompt with63.5% fewer parameters than the old Workspace for only-1.19 points. The likely performance bottleneck is not insufficient module activity but non-complementary control: text-pooled slots remain highly correlated, the visual cross update dominates their query content, and the added ten LLM Workspace tokens become a large second prompt bank. Before adding capacity back, use same-checkpoint scaling/disable controls to separate harmful text write magnitude, visual write magnitude and collapsed Z content; do not rerun an unchanged seed.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_directional_concat_workspace_z10_d1024_l17_private_p20_s8_seed44_20260830`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_test/epoch_3` and `eval_test_epoch_3.log`.

### 2026-08-31 - slake_directional_concat_delta_interventions_seed44_20260831

- Commit/config: commit `05a7ef2`; four inference-only controls reuse the exact Directional Concat Workspace seed44 epoch3 checkpoint and its fixed **77.1729** official-Test predictions. All weights, inputs, decoding settings, `P20+A_t10`, `S8+A_v10`, Prompt30, 18 visual insertion tokens, masks, positions and token counts remain fixed. Only the text or visual `MLP(Z)` dynamic delta is multiplied by `0` or `0.5`; anchors remain present.
- Dataset and split: official SLAKE test.json, 2,094 questions: 836 CLOSED / 1,258 OPEN, 267 KVQA / 1,827 VQA, 1,061 EN / 1,033 ZH.
- Seed / data seed: 44 / 42; same-checkpoint inference interventions with no retraining.
- Reference scale `text=1, visual=1`: Overall **77.17**; CLOSED / OPEN **82.89 / 73.37**; KVQA / VQA **61.05 / 79.53**; EN / ZH **78.32 / 75.99**.
- `text_delta_half`: Overall **76.03**; CLOSED / OPEN **82.66 / 71.62**; KVQA / VQA **59.93 / 78.38**; EN / ZH **77.38 / 74.64**. Effect versus reference **-1.1461**, intervention-only/reference-only correct **26/50**, exact McNemar `p=0.007905`. The effective text delta/anchor ratio is reduced from `0.5045` to `0.2523` while the raw delta remains unchanged.
- `text_delta_off`: Overall **74.26**; CLOSED / OPEN **82.06 / 69.08**; KVQA / VQA **57.68 / 76.68**; EN / ZH **75.97 / 72.51**. Effect **-2.9131**, exclusive correct **38/99**, exact McNemar `p=1.8779e-7`. The loss is concentrated beyond CLOSED questions: OPEN drops **4.29**, KVQA **3.37**, VQA **2.85** and Chinese **3.48** points relative to the reference.
- `visual_delta_half`: Overall **77.22**; CLOSED / OPEN **82.89 / 73.45**; KVQA / VQA **61.42 / 79.53**; EN / ZH **78.23 / 76.19**. Effect **+0.0478**, exclusive correct **7/6**, exact McNemar `p=1.0`. The effective visual delta/anchor ratio is halved from `28.1670` to `14.0835`.
- `visual_delta_off`: Overall **77.13**; CLOSED / OPEN **82.66 / 73.45**; KVQA / VQA **60.30 / 79.58**; EN / ZH **78.13 / 76.09**. Effect **-0.0478**, exclusive correct **17/18**, exact McNemar `p=1.0`. The dynamic visual delta and its `28.1670x` anchor ratio are reduced exactly to zero; private `S8`, static `A_v10`, the visual Block17 insertion interface and the visually conditioned Z-to-text path remain active.
- Diagnostics/control audit: all four controls report anchors and token counts preserved. Raw Z and readout statistics are identical across variants: cross-delta/query `3.3549`, Z slot pairwise cosine `0.9410`, text pooling entropy `0.8630`, visual attention entropy `0.7592`, raw text delta norm `85.7698` and raw visual delta norm `18.2963`. This confirms that only the selected final write magnitude changes. Timing differences are treated as uncontrolled runtime noise because computation is not removed.
- Conclusion: reject the hypothesis that the Directional model underperforms because its text write is too strong. Full-strength Z-to-text writing is monotonically and significantly better than half or zero, supplies **2.91 points** over the anchor-only text interface, and contributes especially to OPEN/KVQA/Chinese performance. Conversely, the learned Z-to-visual dynamic write is causally dispensable at this checkpoint despite its huge norm: half and zero scales are statistically identical to the reference. This does not yet eliminate the complete visual insertion branch because `S8+A_v10` remains. The remaining gap to the 78.37 Full Workspace reference should be sought in Z construction/slot diversity or text-side transformation capacity, not by damping either output; do not run a text-scale sweep.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_directional_concat_delta_interventions_seed44_20260831`; manifest `intervention_manifest.tsv`; per-variant predictions and summaries under `text_delta_half`, `text_delta_off`, `visual_delta_half` and `visual_delta_off`; aggregate paired results `workspace_path_comparison.tsv` and `workspace_path_comparison.json`.

### 2026-08-31 - slake_directional_concat_visual_memory_mismatch_seed44_20260831

- Commit/config: commit `c4e0b7b`; inference-only causal control reusing the exact Directional Concat Workspace seed44 epoch3 checkpoint and its fixed **77.1729** official-Test reference. The original image still follows the complete frozen VLM path unchanged. Only the layer17 visual K/V read by the Directional CA is replaced with the previous distinct image's cached visual Memory; question-derived Q10, all anchors, weights, token counts, positions and decoding settings remain unchanged.
- Dataset and split: official SLAKE test.json, all 2,094 questions. Three excluded timing warmups prime the cache with natural images; the run audits that all **2,094/2,094** scored samples receive K/V from a different image while retaining their correct original image input.
- Seed / data seed: 44 / 42; same-checkpoint intervention with no retraining.
- Controlled change: matched current-image Directional-CA visual K/V -> previous-distinct-image visual K/V. This intervenes only on the sample-conditioned `visual Memory -> Z -> dynamic Prompt` route and does not corrupt the frozen VLM's native visual tokens.
- Official Test Overall: **76.41** (1,600/2,094), versus matched-Memory **77.17** (1,616/2,094): **-0.7641** point.
- CLOSED / OPEN: **82.18 /72.58**, changes **-0.72 /-0.80** from 82.89 /73.37.
- KVQA / VQA: **60.67 /78.71**, changes **-0.37 /-0.82** from 61.05 /79.53.
- EN / ZH: **77.47 /75.31**, changes **-0.85 /-0.68** from 78.32 /75.99.
- Paired test: mismatch-only/reference-only correct counts are **10/26**; exact McNemar `p=0.01133`. Predictions lose a statistically significant net16 correct answers under visual-Memory mismatch.
- Parameter/runtime note: no parameter or training change; the reused model has **12,726,784** trainable parameters. Evaluation TTFT mean0.066899s, weighted TPOT0.019606s/token,51.004 decode tokens/s and request mean0.118297s; timing is descriptive only.
- Intervention audit/diagnostics: `original_image_input_unchanged=True`, `intervention_scope=directional_ca_visual_kv_only`, `anchors_preserved=True`, `token_count_preserved=True` and `visual_memory_mismatch_audit_pass=True`. Mismatched source Memory remains highly similar to the natural source (`cosine=0.9163`), while aggregate Z statistics barely move: slot cosine0.9402, cross-delta/query3.3289 and text delta/anchor0.5062, versus reference0.9410/3.3549/0.5045.
- Conclusion: the compact Directional model is not merely question-conditioned Prompt Tuning. Even though the frozen VLM still receives the correct image and the mismatched medical visual Memories are highly correlated, corrupting only the current-image K/V used to build Z causes a significant0.76-point loss. This establishes a causal but modest image-specific contribution through `visual K/V -> Z -> LLM Prompt`. It does not revive the causally dispensable Z-to-visual dynamic write, prove that visual-encoder tuning is necessary, or attribute the full2.77-point gain over Static Prompt to image conditioning. The result materially strengthens the dynamic multimodal Prompt route while leaving cross-dataset performance and efficiency as its remaining publication risks.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_directional_concat_visual_memory_mismatch_seed44_20260831`; predictions/summary under `visual_kv_previous_distinct_image`; aggregate paired result `workspace_path_comparison.json`; audit manifest `intervention_manifest.tsv`.

### 2026-08-31 - pathvqa_directional_concat_workspace_text_dynamic_only_z10_d1024_l17_private_p20_s8_seed44_20260831

- Commit/config: commit `dbb5771`; compact Directional Concat Workspace retaining learnable private text P20, private visual S8, static visual anchor A_v10, static text anchor A_t10, full-question attention-pooled Q10, layer17 full visual K/V and Directional CA1024/16 producing Z10. Z continues through the zero-initialized text projection into ten LLM Prompt tokens, while the causally dispensable `Z -> visual dynamic MLP` is not instantiated. The visual block still receives static `S8 + A_v10` once and immediately strips all18 tokens after layer17.
- Dataset and split: PathVQA official validation, all6,259 questions and832 image clusters; fixed epoch3 protocol, no Test evaluation.
- Seed / data seed: 44 / 42.
- Controlled change: first PathVQA transfer of the compact Directional model, with only the2,101,248-parameter visual dynamic projection removed from the12,726,784-parameter SLAKE Directional design. P20/A_t10/Q10/visual K/V/Z10/text dynamic write and the complete static visual insertion interface are preserved.
- Validation Overall: **59.5782**; image-clustered95% bootstrap CI **[58.1453,60.9459]**.
- Yes/No / Free-form: **91.5520 /27.6962**.
- Per question type: how11.6279, other7.1429, what22.8022, when7.6923, where65.7702, why4.7619, yes/no91.5520; Free-form question-type macro19.9662.
- Trainable parameters: **10,625,536** total = soft Prompt76,800 + private visual S8,192 + shared Directional/Text modules10,540,544. This is2,101,248 fewer than the original Directional model (-16.51%),86.18% fewer than76,896,256-parameter Full Workspace, and1.501x the7,077,888-parameter full-model Attention LoRA-r8.
- Training: Prompt LR0.3, sparse visual LR3e-5, Directional Workspace LR1e-4, effective batch32,3 epochs; runtime6,510.4s,1,845 steps and train loss11.0995. All requested modules passed the exact optimizer-group and parameter-count audits.
- Diagnostics: Validation means show Z10 slot pairwise cosine0.7483, text pooling normalized entropy0.7723, visual attention normalized entropy0.6202, Cross-Attention delta/query1.3312, text dynamic delta/anchor0.4823 and Z norm27.6647. Static visual insertion is audited active while all visual dynamic-write diagnostics are exactly zero. The final training row remains finite and active: shared Workspace grad norm0.9999, sparse visual grad norm0.00858, text delta/anchor0.5613 and slot cosine0.8945.
- Versus full-model Attention LoRA-r8:59.5782 versus59.3386, only **+0.2397** Overall or15 net correct answers. Current-only/LoRA-only correct counts are336/321, exact McNemar `p=0.5850`, and image-clustered paired delta95% CI is **[-0.5998,+1.1376]**. Free-form is+1.1168 (246/211,`p=0.1116`) while Yes/No is-0.6400 (90/110,`p=0.1790`). The method significantly improves `what` by+3.0220 (`p=5.42e-5`) but significantly loses `where` by-9.5355 (`p=4.32e-5`). It statistically ties LoRA and must not be described as a significant win.
- Versus76.90M Full Workspace: **+0.1278** Overall with323/315 exclusive correct, `p=0.7817`, paired clustered CI[-0.6734,+0.9807]. It preserves statistically indistinguishable accuracy with7.24x fewer parameters, trading+0.9280 Yes/No for-0.6701 Free-form.
- Versus57.5172 single-pass P20+Dynamic-CA+Sparse-Visual baseline: **+2.0610** Overall with382/253 exclusive correct, exact McNemar `p=3.46e-7`, and paired clustered CI **[+1.2862,+2.8909]**. Gains are independently significant on Yes/No (+1.8240,`p=0.000188`) and Free-form (+2.2974,`p=0.000426`).
- Evaluation timing: TTFT mean0.05167s, weighted TPOT0.019871s/token,50.324 decode tokens/s and request mean0.10096s. The Dynamic Prompt timing record counts prepended Prompt positions in `generated_tokens`, so only the explicit latency fields should be used; do not compare its aggregate generated-token throughput directly with LoRA.
- Conclusion: the pre-registered numerical survival rule is met because Overall narrowly exceeds59.3386 without a Free-form decline. The defensible result is not that the method beats LoRA, but that question-conditioned Directional Prompting significantly improves the prior joint baseline and compresses Full Workspace by86.18% without measurable loss. This establishes a strong new Pareto architecture and justifies retaining the route, while the1.50x parameter cost and nonsignificant LoRA difference remain open efficiency risks. Stop unconstrained architecture search; next evidence must be replication and question/image-conditioning mechanism controls rather than another structural redesign.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d1024_l17_private_p20_s8_seed44_20260831`; selected checkpoint `checkpoints/epoch_3`; Validation summary/predictions under `eval_validation/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### PathVQA Yes/No Class-Balance Audit

Correction (2026-08-29): the two `shared_s_memory_main_merger` experiments above did not test the intended raw shared-S architecture. Their implementation captured the sample-conditioned Rep tokens after visual block17, mapped those outputs through `visual.merger`, and fed the result back as a third Text-CA Memory. They therefore reject only a final-anchor Rep feedback shortcut. They do not reject mapping the original, unconditioned shared parameter bank `S` into the LLM prompt space. Two corrected seed44 experiments are pending: (1) retain independent P20 and add a separate zero-initialized CA128 residual over `visual.merger(raw S)` while leaving the winning visual/question Dynamic CA unchanged; (2) remove independent P/P0 entirely and use the two native `visual.merger(raw S)` outputs directly as the prompt and visual/question CA queries. Neither corrected method reads any Rep output from layers5/11/17.

| Method | Yes count | Yes accuracy | No count | No accuracy | Class gap |
|---|---:|---:|---:|---:|---:|
| Frozen Base | 1,816 | 53.30 | 1,546 | 83.83 | 30.53 |
| Earlier 128-slot MMRL seed44 | 1,816 | 80.62 | 1,546 | 84.86 | 4.24 |
| Minimal MMRL Relation0 seed44 | 1,816 | 69.11 | 1,546 | 90.88 | 21.77 |
| Minimal MMRL Relation0.05 seed44 | 1,816 | 79.68 | 1,546 | 86.03 | 6.35 |
| Visual LoRA-r128 seed44 | 1,816 | 83.65 | 1,546 | 89.39 | 5.74 |
| Static Prompt Tuning seed44 | 1,816 | 91.30 | 1,546 | 89.20 | 2.10 |
| Dynamic Multimodal Prompt seed44 | 1,816 | 93.72 | 1,546 | 85.25 | 8.47 |

The aggregate Yes/No score hides a severe Base bias toward `no`. MMRL removes most of this imbalance and therefore does not obtain82.57 by collapsing to one class. It nevertheless trails the all24-layer LoRA run by3.03 points on `yes` and4.53 points on `no`. Same-question paired exact McNemar tests reject a tie for this unmatched comparison: on `yes`, MMRL-only/LoRA-only correct counts are127/182 (`p=0.0021`); on `no`,64/134 (`p=7.29e-7`). These statistics remain valid for the completed models but cannot establish a scope-matched architectural advantage because MMRL acts on visual layers17-24 while this LoRA acts on all24 layers.

Correction (2026-08-27): the original `MMRL seed44` class row above refers to the earlier 128-slot47.30 run. The two minimal-MMRL rows were added from their own stored comparison files and are the correct class breakdowns for the clean Relation experiment. They show that Relation's binary gain is primarily calibration of affirmative answers rather than a uniform visual-accuracy increase.

## Rejected / Superseded Directions

- DeepStack residual injection: large regression to68.53.
- Mixed Cross-Relation and Cross-Relation-only: no improvement over the clean reference.
- Reverse assignment and all dynamic-query reversal variants: 61.70-70.96, all below reference.
- Competitive and dual-softmax visual assignment: attention remained nearly uniform.
- Balanced text-guided fusion: slot collapse and70.49.
- Alpha gate/Relation weighting: no stable multi-seed gain.
- Learned 128-slot Attention Pooling: no mean gain over Mean Pooling on SLAKE.

These results remain useful negative evidence and should not be rerun unless a new hypothesis specifically addresses the observed failure mechanism.

## Pending Experiments

1. Fair CA replacement: separately normalize Q, visual memory, and text memory before the equal-parameter Concat-MLP; run seed45 only.
2. Add a supplementary semantic metric or blinded manual sample for PathVQA Free-form answers while retaining normalized exact-match as the official primary metric.
3. Confirm the clean PathVQA Relation gain on one additional seed before making a cross-seed claim; the seed44 paired result is already statistically clear.
4. Final comparison table: Base VLM, Visual LoRA, Static Prompt Tuning, nearest visual prompt/adapter baseline, and MMRL.

## Update Template

```markdown
### YYYY-MM-DD - experiment_name

- Commit/config:
- Dataset and split:
- Seed / data seed:
- Controlled change:
- Overall:
- CLOSED / OPEN:
- KVQA / VQA:
- EN / ZH:
- Trainable parameters:
- Runtime:
- Diagnostics:
- Conclusion:
- Output/log path:
```

### 2026-08-31 - pathvqa_directional_concat_conditioning_mismatch_seed44_20260831_1

- Commit/config: inference controls and strict audits at `f1c12b7`; fixed trained checkpoint `pathvqa_directional_concat_workspace_text_dynamic_only_z10_d1024_l17_private_p20_s8_seed44_20260831/checkpoints/epoch_3`, 10,625,536 trainable parameters during its original training. No retraining occurred for either control.
- Dataset and split: PathVQA full Validation, 6,259 questions grouped into 832 decoded-image clusters.
- Seed / data seed: checkpoint seed44 / data seed42; paired bootstrap seed42 with10,000 iterations.
- Controlled changes: the correct image and question remain on the complete native VLM path. `question_q_previous_distinct` changes only the Directional CA `Q10` to the previous distinct question's pooled query while retaining the current visual K/V. `visual_kv_previous_distinct_image` retains the current `Q10` and changes only Directional CA visual K/V to the previous distinct image. Each run uses one excluded priming inference whose question and image both differ from the first scored sample.
- Fixed matched baseline: **59.5782 Overall / 91.5520 Yes/No / 27.6962 Free-form**.
- Question-Q mismatch: **49.9441 Overall / 89.2800 Yes/No / 10.7211 Free-form**, deltas **-9.6341 / -2.2720 / -16.9751**. Mismatch-only/reference-only correct counts are111/714, exact McNemar `p=1.37e-108`; image-clustered paired delta95% CI **[-10.5451,-8.7203]**. Question-type deltas are how-3.1008, other0, what-13.8540, when0, where-42.5428, why-4.7619 and yes/no-2.2720.
- Visual-K/V mismatch: **52.5483 Overall / 86.2720 Yes/No / 18.9215 Free-form**, deltas **-7.0299 / -5.2800 / -8.7747**. Mismatch-only/reference-only correct counts are151/591, exact McNemar `p=2.70e-62`; image-clustered paired delta95% CI **[-7.9508,-6.1191]**. Question-type deltas are how-0.7752, other-7.1429, what-7.4568, when0, where-20.2934, why0 and yes/no-5.2800.
- Intervention audit: both runs preserve original image input, original question input, anchors and token count. Each records exactly6,260 conditioning invocations: one excluded natural prime and **6,259/6,259 mismatched scored samples**. Mode propagation and mismatch audits pass. Mean substituted-source cosine is0.4804 for question `Q10` and0.7613 for visual K/V; therefore the larger Question-Q score loss must not be interpreted as a calibrated importance comparison because the two corruptions have different semantic severity.
- Diagnostics: under Question-Q mismatch, workspace cross-delta/query is1.3313, slot cosine0.7496 and text delta/anchor0.4708. Under visual-K/V mismatch they are1.4749,0.7811 and0.4845. Both corruptions leave a large, active dynamic prefix rather than turning the branch off; their losses establish the necessity of correct cross-modal alignment, not the standalone gain magnitude of either input.
- Conclusion: the compact Directional model is neither static Prompt Tuning nor a question-only/image-only shortcut. Its performance causally depends on matching the current question-derived query with the current image's frozen visual features before writing the LLM prompt. The strongest effect appears on Free-form and spatial `where` questions, directly supporting the paper mechanism of on-demand question-guided visual evidence retrieval. Because mismatched evidence can be actively harmful, the9.63/7.03-point losses cannot be reported as additive module contributions. Together with the prior no-effect dynamic visual-write intervention, these controls support a read-only-vision/write-only-language design; they do not yet remove the remaining static visual prompts or prove that all visual-side calibration is unnecessary.
- Initial failed invocation: `pathvqa_directional_concat_conditioning_mismatch_seed44_20260831` stopped before any model evaluation because a legacy unit test expected the old visual-only scope label. Commit `f1c12b7` restored mode-specific scope reporting; the failed directory contains no experimental predictions and is not a model result.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_conditioning_mismatch_seed44_20260831_1`; paired outputs `conditioning_mismatch_comparison.{json,tsv}`, per-control summaries and predictions under `question_q_previous_distinct/` and `visual_kv_previous_distinct_image/`.

### 2026-09-01 - pathvqa_directional_concat_workspace_text_dynamic_only_z10_d512_l17_private_p20_s8_seed44_20260831

- Commit/config: commit `b7c1003`; exact 512-dimensional compression of the locked Directional Text-Dynamic-Only architecture. P20, static visual `S8+A_v10`, static text anchor A_t10, attention-pooled question Q10, full layer17 visual K/V, Z10, CA16 heads, `Z -> LLM Prompt`, all learning rates and the read-only-vision/write-only-language interface are unchanged. Only the Directional latent width changes from1024 to512: text projection `2560->512`, visual projection `1024->512`, CA operates at512 dimensions and the text output path is `512->512->2560`.
- Dataset and split: PathVQA official Validation, all6,259 questions and832 image clusters; fixed epoch3 protocol, no Test evaluation.
- Seed / data seed: 44 / 42; paired bootstrap seed42 with10,000 iterations.
- Controlled change: Directional latent width `1024 -> 512`; no slot-count, layer, optimizer, data-order or output-interface change.
- Validation Overall: **58.6835**; image-clustered standalone95% bootstrap CI **[57.1373,60.0827]**.
- Yes/No / Free-form: **90.8480 /26.6114**. Yes-reference / No-reference accuracy is92.4065 /88.9597.
- Per question type: how10.8527, other7.1429, what21.6248, when0.0000, where65.2812, why4.7619, yes/no90.8480; Free-form question-type macro18.2772.
- Trainable parameters: **4,591,616** total = soft Prompt76,800 + private visual S8,192 + shared Directional/Text modules4,506,624. This is6,033,920 / **56.79% fewer** than the10,625,536-parameter D1024 reference and2,486,272 / **35.13% fewer** than the7,077,888-parameter full-model Attention LoRA-r8.
- Training: Prompt LR0.3, sparse visual LR3e-5, Directional LR1e-4, effective batch32,3 epochs and1,845 steps; runtime6,574.16s, reported train loss11.6124. Width compression does not reduce end-to-end training time versus D1024's6,510.4s because the frozen VLM forward/backward path dominates.
- Versus D1024: **-0.8947 Overall**, with D512-only/D1024-only correct counts240/296, exact McNemar `p=0.01744` and image-clustered paired95% CI **[-1.6651,-0.1275]**. The loss is statistically significant but remains inside the pre-registered0.5-1.5-point acceptable compression band. Yes/No changes-0.7040 (90/112,`p=0.1393`) and Free-form-1.0849 (150/184,`p=0.07081`); `what` changes-1.1774 (`p=0.05561`) and `where`-0.4890 (`p=0.9196`). Normalized predictions remain identical on4,505/6,259 questions.
- Versus LoRA-r8: **-0.6551 Overall**, D512-only/LoRA-only correct319/360, exact McNemar `p=0.1247`, paired clustered95% CI **[-1.5418,+0.2234]**. Overall remains statistically tied while D512 uses35.13% fewer parameters. Free-form is effectively equal at+0.0319 (234/233,`p=1.0`) and `what` is significantly better by+1.8446 (`p=0.01479`), but Yes/No is significantly worse by-1.3440 (`p=0.00475`) and `where` by-10.0244 (`p=3.11e-5`). The binary deficit versus LoRA is concentrated on `yes` references (-1.9276,`p=0.000610`) rather than `no` (-0.6369,`p=0.4709`).
- Diagnostics: all requested gradients remain finite and active; final shared Workspace grad is0.9993 and sparse visual grad0.03283. Relative to D1024 validation means, D512 has higher slot cosine0.8011 versus0.7483, more diffuse visual attention entropy0.7557 versus0.6202, weaker Cross-Attention delta/query0.9800 versus1.3312 and weaker text delta/anchor0.3840 versus0.4823. Question-pooling entropy falls0.7723 ->0.5895 and Z norm stays similar26.58 versus27.66. Together with the higher train loss, this is consistent with a real capacity bottleneck and weaker visual-to-prompt transfer, not branch collapse or optimizer failure.
- Evaluation timing: TTFT mean0.052686s, weighted TPOT0.019562s/token,51.119 decode tokens/s and request mean0.104388s. There is no demonstrated latency gain over D1024; timing differences are session noise and the frozen backbone dominates.
- Conclusion: D512 is **not** an accuracy-preserving replacement for D1024; the approximately0.9-point loss is paired-significant, so D1024 remains the primary accuracy configuration. It is nevertheless a valid Pareto model: a56.79% parameter reduction for less than one point, and35.13% fewer trainable parameters than LoRA-r8 while retaining statistically tied Overall and equal Free-form. Do not scan D256/D768 immediately. First decide whether the paper should report D1024 as the main model and D512 as an efficiency ablation; any further slimming must address slot diversity and cross-modal transfer rather than merely reducing width again.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d512_l17_private_p20_s8_seed44_20260831`; selected checkpoint `checkpoints/epoch_3`; Validation summary/predictions under `eval_validation/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; training report `train_report.json`.

### 2026-09-01 - pathvqa_directional_concat_workspace_text_dynamic_only_z10_d768_l17_private_p20_s8_seed44_20260901

- Commit/config: commit `48383be`; locked Directional Text-Dynamic-Only architecture with latent width768. P20, static visual `S8+A_v10`, static text anchor A_t10, attention-pooled Q10, full layer17 visual K/V, Z10, CA16 heads, read-only-vision/write-only-language output, learning rates and training protocol are identical to D256/D512/D1024.
- Dataset and split: PathVQA official Validation, all6,259 questions and832 image clusters; fixed epoch3 protocol, no Test evaluation.
- Seed / data seed: 44 / 42; paired bootstrap seed42 with10,000 iterations.
- Controlled change: Directional latent width `1024 -> 768`; text projection `2560->768`, visual projection `1024->768`, CA768/16 and text output `768->768->2560`. No other architecture, optimizer, data-order or interface change.
- Validation Overall: **59.5622**; image-clustered standalone95% bootstrap CI **[58.0759,60.9533]**.
- Yes/No / Free-form: **91.0400 /28.1749**. Yes-reference / No-reference accuracy is94.1005 /87.3319.
- Per question type: how12.4031, other7.1429, what22.2920, when0.0000, where72.6161, why4.7619, yes/no91.0400; Free-form question-type macro19.8693.
- Trainable parameters: **7,805,184** total = soft Prompt76,800 + private visual S8,192 + shared Directional/Text modules7,720,192. This is2,820,352 / **26.54% fewer** than D1024 and727,296 /10.28% more than full-model Attention LoRA-r8.
- Training: Prompt LR0.3, sparse visual LR3e-5, Directional LR1e-4, effective batch32,3 epochs and1,845 steps; runtime6,440.81s and train loss11.2930. Runtime is not materially different from D1024/D512/D256 because frozen-backbone computation dominates.
- Versus D1024: **-0.0160 Overall**, D768-only/D1024-only correct265/266, exact McNemar `p=1.0` and image-clustered paired95% CI **[-0.7630,+0.7387]**. Free-form changes+0.4786 (185/170,`p=0.4575`) while Yes/No changes-0.5120 (80/96,`p=0.2581`). `where` improves significantly by+6.8460 (51/23,`p=0.001516`), while `what` changes-0.5102 without significance. Normalized predictions remain identical on4,484/6,259 questions. D768 is accuracy-equivalent to D1024, not merely numerically close.
- Versus D512: **+0.8787 Overall**, D768-only/D512-only correct277/222, exact McNemar `p=0.01555` and paired clustered95% CI **[+0.1285,+1.5969]**. Free-form improves significantly by+1.5635 (`p=0.006748`), Yes/No changes only+0.1920 (`p=0.7125`) and `where` improves+7.3350 (`p=0.000358`). The extra width recovers a real open/spatial capability rather than only binary calibration.
- Versus LoRA-r8: **+0.2237 Overall**, D768-only/LoRA-only correct332/318, exact McNemar `p=0.6102` and paired clustered95% CI **[-0.6712,+1.1196]**. Overall is statistically tied. D768 significantly improves Free-form by+1.5954 (`p=0.01967`) and `what` by+2.5118 (`p=0.000904`), while LoRA significantly leads Yes/No by1.1520 (`p=0.01503`); the `where` difference-2.6895 is no longer significant (`p=0.2215`).
- Diagnostics: all gradients remain finite and active; final shared Workspace grad is0.9998. Validation means are slot cosine0.7118, text-pooling entropy0.7472, visual-attention entropy0.7133, Cross-Attention delta/query1.1751, text delta/anchor0.4987 and Z norm24.9050. Relative to D1024, slot diversity and text write strength are preserved while visual attention is moderately more diffuse; relative to D512, slot cosine, attention selectivity and text write all recover. This matches the restored score and rules out a width-independent optimization ceiling.
- Evaluation timing: TTFT mean0.051219s, weighted TPOT0.019662s/token,50.859 decode tokens/s and request mean0.104715s. Timing differences across widths are descriptive runtime noise, not a claimed speedup.
- Conclusion: D768 is the preferred **high-performance configuration**. It removes26.54% of D1024 trainable parameters with no measurable Overall loss, raises Free-form numerically and sharply improves `where`. Against LoRA-r8 it remains Overall-tied but has a significant Free-form/`what` advantage at10.28% more parameters. D1024 is therefore a capacity upper endpoint rather than the best default; D512 remains the below-LoRA-parameter efficiency point.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d768_l17_private_p20_s8_seed44_20260901`; selected checkpoint `checkpoints/epoch_3`; Validation under `eval_validation/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; training report `train_report.json`.

### 2026-09-02 - PathVQA QDPT-D768 multi-seed replication (seeds45-46)

- Exact experiment: `pathvqa_directional_concat_workspace_text_dynamic_only_z10_d768_l17_private_p20_s8`; same locked D768 architecture and training protocol as seed44, with only model/training seed changed. Dataset/data seed remain PathVQA official Validation /42; each evaluation covers all6,259 questions and832 image clusters at epoch3 only.
- Seed45: Overall **58.6835**, standalone image-clustered95% CI **[57.2405,60.0510]**; Yes/No / Free-form **90.3360 /27.1219**; how/other/what/when/where/why **13.1783/7.1429/21.6248/0.0000/68.7042/0.0000**. Trainable parameter audit **7,805,184**; runtime6,473.07s,1,845 steps, train loss11.38585. Diagnostics: slot cosine0.7441, question-pooling entropy0.6951, visual-attention entropy0.5695, Cross-Attention delta/query1.5016, text delta/anchor0.4483 and Z norm32.7412.
- Seed46: Overall **58.7634**, standalone image-clustered95% CI **[57.3452,60.1045]**; Yes/No / Free-form **92.2560 /25.3669**; how/other/what/when/where/why **10.8527/14.2857/20.4082/0.0000/63.0807/4.7619**. Trainable parameter audit **7,805,184**; runtime6,439.31s,1,845 steps, train loss11.17069. Diagnostics: slot cosine0.8830, question-pooling entropy0.8015, visual-attention entropy0.6308, Cross-Attention delta/query1.6430, text delta/anchor0.5208 and Z norm27.7628.
- Three-seed summary (44/45/46): Overall **59.0030 +/- 0.4859** sample standard deviation; Yes/No **91.2107 +/- 0.9709**; Free-form **26.8879 +/- 1.4190**. Seed44's59.5622 is the high endpoint rather than the multi-seed center. Seed46 exchanges lower Free-form for the strongest Yes/No, so seed variation primarily changes capability balance rather than causing global collapse.
- Conclusion: the replication is necessary and not wasted. It replaces a single-seed59.56 claim with a defensible approximately59.00 multi-seed estimate and exposes nontrivial open-answer variance. Final comparisons must use matched QDPT/LoRA seeds or multi-seed means; do not compare seed44 QDPT against future seed45/46 LoRA as if seed were controlled.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d768_l17_private_p20_s8_seed45_20260902` and `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d768_l17_private_p20_s8_seed46_20260902`; selected checkpoints `checkpoints/epoch_3`, summaries under `eval_validation/epoch_3`, diagnostics `dynamic_prompt_diagnostics.jsonl` and reports `train_report.json`.

### 2026-09-03 - PathVQA QDPT-D768 direct-visual-Z concatenation (seeds44-46)

- Commit/config: implementation `1aefeb2`, three-seed suite `c4c7247`; `pathvqa_qdpt_d768_question_q10_l17_p20_av10_direct_zv10`. The locked D768 question-Q/visual-KV Directional CA and text path remain unchanged. The Layer17 visual prefix changes from the reference `[S_v8; A_v10]` to `[A_v10; LayerNorm+Linear(Z10)]`, removing the private `S_v8` and directly projecting the shared conditional `Z10` from768 to1024 dimensions. The20-token prefix passes through frozen Visual Block17 once and is then stripped.
- Dataset and split: PathVQA official Validation, all6,259 questions and832 image clusters; only epoch3 is evaluated and Test is not run.
- Seed / data seed: model/training seeds44,45,46 / fixed data and bootstrap seed42.
- Controlled change: direct conditional `Z10` replaces private static `S_v8` in the visual prefix; prompt lengths, D768, Q10, A_v10, P20/A_t10, optimizer, data order and text-side dynamic Prompt path are otherwise fixed.
- Seed44: Overall **59.6741**, image-clustered95% CI **[58.2130,61.0742]**; Yes/No / Free-form **90.8800 /28.5578**; how/other/what/when/where/why **11.6279/14.2857/22.9592/0.0000/71.3936/4.7619**.
- Seed45: Overall **57.9006**, image-clustered95% CI **[56.4549,59.2945]**; Yes/No / Free-form **90.0480 /25.8456**; how/other/what/when/where/why **9.3023/7.1429/21.1931/0.0000/62.5917/4.7619**.
- Seed46: Overall **58.8273**, image-clustered95% CI **[57.3918,60.2021]**; Yes/No / Free-form **90.4960 /27.2495**; how/other/what/when/where/why **10.8527/14.2857/22.1350/7.6923/66.5037/4.7619**.
- Three-seed summary: Overall **58.8007 +/- 0.8870**, Yes/No **90.4747 +/- 0.4164**, Free-form **27.2176 +/- 1.3564** (mean +/- sample standard deviation).
- Matched-seed comparison with retained-static D768 `[S_v8; A_v10]`: Overall deltas are **+0.1119/-0.7829/+0.0639** for seeds44/45/46, with mean delta **-0.2023**. Mean Yes/No changes **-0.7360**, while mean Free-form changes **+0.3297**. Overall variability increases from0.4859 to0.8870. Thus the two positive per-seed differences are negligible and do not compensate for seed45's loss.
- Trainable parameters: **8,585,984** total for every seed = soft Prompt76,800 + shared Directional/direct-Z modules8,509,184. This is780,800 more than the7,805,184-parameter reference D768.
- Evaluation diagnostics: direct visual Z is active rather than bypassed. Across seeds44/45/46, its mean norm is32.82/34.17/32.33, visual-prefix/input ratio0.528/0.549/0.521 and Visual Block output/input ratio1.166/1.169/1.166. Slot pairwise cosine is0.609/0.702/0.762, question-pooling entropy0.698/0.714/0.584, visual-attention entropy0.749/0.755/0.692, CA delta/query1.007/1.132/0.840 and text delta/anchor0.392/0.430/0.391. The direct-Z/static-anchor norm ratio near50x reflects the deliberately tiny A_v10 initialization and must not be interpreted as a50x effect on the image tokens.
- Conclusion: direct `Proj(Z10)` successfully and substantially perturbs Layer17, but stronger conditional visual writing does **not** produce a stable accuracy gain. Compared with the simpler retained-static D768 it loses0.20 points on the matched three-seed mean, costs0.78M additional parameters and nearly doubles Overall seed standard deviation. Together with the question/visual-KV mismatch and no-static-visual controls, this supports the bounded claim that, in the current frozen generative MLLM setting, question conditioning is most useful for post-encoding evidence retrieval and LLM-side interpretation: the frozen encoder retains broadly useful evidence, the static visual Prompt supplies stable domain calibration, and sample-conditioned Z need not rewrite the visual stream. Direct visual-Z concatenation is rejected as the final architecture; this finding must not be generalized to all models or tasks.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_av10_direct_zv10_seed{44,45,46}_20260902`; selected checkpoints `checkpoints/epoch_3`, summaries under `eval_validation/epoch_3`, diagnostics `dynamic_prompt_diagnostics.jsonl` and reports `train_report.json`.

### 2026-09-01 - pathvqa_directional_concat_workspace_text_dynamic_only_z10_d256_l17_private_p20_s8_seed44_20260901

- Commit/config: commit `48383be`; exact D256 endpoint of the same locked Directional Text-Dynamic-Only width ablation. All non-width settings are identical to D512/D768/D1024.
- Dataset and split: PathVQA official Validation, all6,259 questions and832 image clusters; fixed epoch3 protocol, no Test evaluation.
- Seed / data seed: 44 / 42; paired bootstrap seed42 with10,000 iterations.
- Controlled change: Directional latent width `1024 -> 256`; text projection `2560->256`, visual projection `1024->256`, CA256/16 and text output `256->256->2560`. No other change.
- Validation Overall: **57.1497**; image-clustered standalone95% bootstrap CI **[55.6468,58.5343]**.
- Yes/No / Free-form: **89.4720 /24.9202**. Yes-reference / No-reference accuracy is92.4065 /85.9165.
- Per question type: how10.8527, other7.1429, what20.1334, when0.0000, where61.6137, why4.7619, yes/no89.4720; Free-form question-type macro17.4174.
- Trainable parameters: **2,033,408** total = soft Prompt76,800 + private visual S8,192 + shared Directional/Text modules1,948,416. This is8,592,128 / **80.86% fewer** than D1024 and5,044,480 / **71.27% fewer** than LoRA-r8.
- Training: Prompt LR0.3, sparse visual LR3e-5, Directional LR1e-4, effective batch32,3 epochs and1,845 steps; runtime6,439.22s and train loss11.7079. Width reduction again produces no material training-time reduction.
- Versus D512: **-1.5338 Overall**, D256-only/D512-only correct239/335, exact McNemar `p=7.07e-5` and paired clustered95% CI **[-2.3380,-0.7400]**. Both Free-form (-1.6911,`p=0.00530`) and Yes/No (-1.3760,`p=0.00500`) decline significantly; `what` also falls-1.4914 (`p=0.01767`). Normalized predictions remain identical on4,403/6,259 questions.
- Versus D1024: **-2.4285 Overall**, D256-only/D1024-only correct232/384, exact McNemar `p=9.70e-10` and paired clustered95% CI **[-3.2863,-1.5665]**. Free-form falls-2.7760 (`p=8.68e-6`), Yes/No-2.0800 (`p=3.13e-5`), `what`-2.6688 (`p=3.64e-5`) and `where`-4.1565 (`p=0.1215`). D256 is a clear low-capacity failure point rather than a noisy efficiency tradeoff.
- Diagnostics: training remains numerically healthy; final shared Workspace grad is0.9992 and sparse visual grad0.02060. Validation slot cosine0.6198 is the lowest and therefore does not indicate slot collapse. Instead, visual-attention entropy rises to **0.9743** (near-uniform), Cross-Attention delta/query falls to **0.2704**, text delta/anchor falls to0.2669 and Z norm to17.3946. The failure mechanism is an information bottleneck in the256-dimensional visual K/V projection and weak cross-modal transfer, despite diverse slots and active gradients.
- Evaluation timing: TTFT mean0.051371s, weighted TPOT0.019723s/token,50.702 decode tokens/s and request mean0.098197s. Lower request mean is descriptive only; no controlled speed claim is made.
- Conclusion: D256 establishes the lower capacity boundary. It saves parameters aggressively but significantly harms both binary and open questions, and should appear only as the lowest-width ablation. Do not continue to D128 or add intermediate widths: the fixed four-point table already shows a plateau at768-1024, a tolerable D512 tradeoff and a D256 cliff.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d256_l17_private_p20_s8_seed44_20260901`; selected checkpoint `checkpoints/epoch_3`; Validation under `eval_validation/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; training report `train_report.json`.

## Provisional Training Resource Observations

### 2026-09-03 - PathVQA unified V20 Layer17 diagnostic

- Exact experiment: `pathvqa_qdpt_d768_question_q10_l17_p20_unified_v20_seed44`; PathVQA official Validation, seed/data seed44/42, epoch3-only full evaluation.
- Intended change: replace the historical two-table `[S_v8; A_v10]` static visual prefix with one `V20 x 1024` table at Layer17, using the Directional/workspace LR1e-4. Q10, D768, full Layer17 visual K/V, P20/A_t10 and the no-dynamic-visual-write policy were intended to remain fixed.
- Result: Overall **56.4291**, image-clustered95% CI **[55.04,57.80]**; Yes/No **89.4400**, Free-form **23.5227**. Relative to the historical seed44 D768 result59.5622 this is -3.1331 Overall, -1.6000 Yes/No and -4.6522 Free-form.
- Trainable parameters: **7,807,232** = soft/text Prompt76,800 + shared Directional/static-visual modules7,730,432. Runtime6,620.40s,1,845 steps and train loss11.29808, essentially equal to the historical train loss11.29295.
- Visual optimization did not collapse or explode. Unified V20 mean norm changes only0.64305 ->0.64980; historical S8 changes0.64033 ->0.64080 and A_v10 changes0.64758 ->0.65534. Thus the loss is not evidence that LR1e-4 directly overtrained V20.
- Initialization audit reveals a confound before any optimizer step: although static text-Prompt and text-anchor initial norms match exactly, question-query norm changes5.46447 ->5.15116, question-pooling entropy0.94745 ->0.95912 and workspace slot cosine0.85547 ->0.91016. The visual Prompt parameters are initialized before the question projection and Directional CA; changing18 sampled rows across two tensors into20 rows in one tensor advances the global RNG stream and changes downstream cross-modal initialization. The run therefore does **not** isolate prefix unification or token count.
- Conclusion: do not use this score to reject a unified static visual Prompt, and do not compare Layer18/multi-layer variants against the historical Layer17 result. First decouple module initialization RNG or otherwise guarantee identical downstream initialization; then rerun the unified Layer17 control before any layer sensitivity experiment.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_unified_v20_seed44_20260903_2`; checkpoint `checkpoints/epoch_3`, Validation under `eval_validation/epoch_3`, diagnostics `dynamic_prompt_diagnostics.jsonl`, report `train_report.json`.

### 2026-09-03 - PathVQA unified V20 Layer17 RNG-controlled rerun

- Exact experiment: `pathvqa_qdpt_d768_question_q10_l17_p20_unified_v20_rng_control_seed44`; commit `4467ee6`; PathVQA official Validation, all6,259 questions and832 image clusters, seed/data seed44/42, epoch3-only full evaluation.
- Controlled change: rerun the unified `V20 x 1024` Layer17 static visual Prompt while consuming the same two historical `A_v10` then `S_v8` random draws before initializing Question Projection and Directional CA. The downstream step0 audit exactly matches the retained `[S_v8; A_v10]` seed44 baseline: question-query norm5.464471, text-pooling entropy0.947449, visual-attention entropy0.995510, Cross-Attention delta norm3.705434, slot cosine0.855469 and visual-memory norm6.564379. The unified table remains one20-token parameter group at LR1e-4; the retained baseline uses8 tokens at3e-5 plus10 at1e-4.
- Validation Overall: **58.5237**, standalone image-clustered95% CI **[57.0216,59.8379]**. Yes/No / Free-form: **90.6880 /26.4518**. Per question type: how9.3023, other7.1429, what21.1931, when0.0000, where67.2372, why4.7619, yes/no90.6880; Free-form question-type macro18.2729.
- Trainable parameters: **7,807,232**. Training runtime6,653.36s,1,845 steps and train loss11.50150.
- Versus the uncorrected unified-V20 run: Overall improves **+2.0946**, Yes/No +1.2480 and Free-form +2.9291. Therefore the majority of its original3.1331-point deficit was caused by downstream initialization drift, not V20 length or optimizer collapse.
- Versus the exact retained `[S_v8; A_v10]` seed44 baseline: Overall remains **-1.0385**; V20-only/baseline-only correct counts251/316, exact McNemar `p=0.007141`, image-clustered paired95% CI **[-1.7844,-0.2075]** from1,000 bootstrap iterations. Free-form falls **-1.7230** (152/206, `p=0.005022`), whereas Yes/No changes only-0.3520 (99/110, `p=0.4892`). The residual loss is therefore paired-significant and concentrated in open answers rather than binary calibration.
- Diagnostics: final training-step unified V20 norm0.64958, soft-Prompt norm625.73, text-anchor norm555.64, text delta/anchor0.54868, Workspace norm49.81, Cross-Attention delta/query2.04756, slot cosine0.80859 and visual-attention entropy0.60753. Validation means are slot cosine0.59365, visual-attention entropy0.62815, Cross-Attention delta/query1.16627, text delta/anchor0.52336 and Workspace norm33.16466. The branch is active and finite; after identical initialization, unified V20 still follows a different optimization trajectory from the dual-rate baseline.
- Conclusion/correction: initialization coupling explains about two thirds of the original numerical collapse, so the earlier56.43 score must not be attributed directly to prefix unification. It does **not** explain the full difference: the controlled V20 rerun still has a statistically supported1.04-point Overall and1.72-point Free-form deficit. Per the predeclared stopping rule, retain historical `S8@3e-5 + A_v10@1e-4` as the final architecture, keep V20 only as a negative ablation, and run any remaining layer-sensitivity or cross-dataset experiments with the retained parameterization.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_unified_v20_rng_control_seed44_20260903`; checkpoint `checkpoints/epoch_3`, Validation under `eval_validation/epoch_3`, diagnostics `dynamic_prompt_diagnostics.jsonl`, report `train_report.json`.

### 2026-09-03 - PathVQA layer-sensitivity evaluation loading failure

- Exact experiments: `pathvqa_qdpt_d768_question_q10_l18_p20_s8_av10_seed44` and `pathvqa_qdpt_d768_question_q10_l17_18_19_shared_p20_s8_av10_seed44`; retained `S8@3e-5+A_v10@1e-4` D768 architecture, seed/data seed44/42,3 epochs and epoch3-only Validation protocol.
- Training completed successfully for both experiments. Layer18-only saved a complete epoch3 checkpoint with7,805,184 trainable parameters after1,845 steps, runtime6,526.10s and train loss11.27955. Shared Layer17+18+19 saved a complete epoch3 checkpoint with the same7,805,184 shared parameters after1,845 steps, runtime6,741.31s and train loss11.38814.
- Evaluation failed before inference for both checkpoints. The loader selected the always-present legacy-compatible field `static_visual_prompt_tokens: 0` instead of `private_visual_prompt_tokens: 8` whenever `directional_concat_workspace` existed. It therefore reconstructed `private_prompt_tokens=0` together with `static_visual_write=true` and raised `ValueError: private_prompt_tokens must be positive when static visual write is enabled`.
- Conclusion: this is a checkpoint deserialization regression introduced by unified-V20 support, not a training failure or model result. Both trained checkpoints remain valid and must be evaluated in place after correcting field selection; do not retrain either experiment. No accuracy is available yet.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l18_p20_s8_av10_seed44_20260903` and `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_18_19_shared_p20_s8_av10_seed44_20260903`.

### 2026-09-04 - PathVQA QDPT-D768 layer sensitivity resolution

- Exact experiments: `pathvqa_qdpt_d768_question_q10_l18_p20_s8_av10_seed44` and `pathvqa_qdpt_d768_question_q10_l17_18_19_shared_p20_s8_av10_seed44`; loader fix commits `835eaeb` and `f23f2c2`. Both reuse their existing epoch3 checkpoints after the evaluation-only failure above. Dataset/data seed remain PathVQA official Validation /42, model seed44, all6,259 questions and832 image clusters, with no Test evaluation.
- Controlled change: the retained D768 `S8@3e-5+A_v10@1e-4` architecture, parameter count7,805,184, optimizer, data order and three-epoch protocol remain fixed. Relative to the Layer17 reference, Layer18-only moves the single shared static-visual/Directional anchor from17 to18. The shared Layer17+18+19 variant applies the same shared parameters at all three layers and uses the final Layer19 workspace for the LLM Prompt; it does not allocate three independent adapters.
- Layer18-only: Overall **58.3959**, standalone image-clustered95% CI **[57.0012,59.7001]**; Yes/No / Free-form **91.3280 /25.5584**. Per question type: how10.8527, other7.1429, what19.6625, when0.0000, where69.4377, why4.7619, yes/no91.3280. Training runtime6,526.10s,1,845 steps and train loss11.27955.
- Layer18-only versus Layer17 reference59.5622: **-1.1663 Overall**, Layer18-only/Layer17-only correct239/312, exact McNemar `p=0.002132`, image-clustered paired95% CI **[-1.9252,-0.4212]** with10,000 iterations. Yes/No changes+0.2880 (103/94, `p=0.5688`), while Free-form falls **-2.6165** (136/218, `p=1.54e-5`); `what` falls-2.6295 (`p=9.68e-5`) and `where`-3.1785 (`p=0.1175`).
- Shared Layer17+18+19: Overall **58.3799**, standalone image-clustered95% CI **[56.8590,59.7839]**; Yes/No / Free-form **91.4560 /25.3989**. Per question type: how6.9767, other14.2857, what20.4867, when0.0000, where64.0587, why4.7619, yes/no91.4560. Training runtime6,741.31s,1,845 steps and train loss11.38814.
- Shared Layer17+18+19 versus Layer17 reference: **-1.1823 Overall**, multi-layer/Layer17-only correct246/320, exact McNemar `p=0.002125`, image-clustered paired95% CI **[-1.9729,-0.3846]**. Yes/No changes+0.4160 (109/96, `p=0.4020`), while Free-form falls **-2.7760** (137/224, `p=5.44e-6`); `what` falls-1.8053 (`p=0.00790`), `where`-8.5575 (`p=1.57e-5`) and `how`-5.4264 (`p=0.01563`).
- Multi-layer versus Layer18-only is a complete Overall tie: **-0.0160**, multi-layer/Layer18-only correct290/291, exact McNemar `p=1.0`, image-clustered paired95% CI **[-0.8214,+0.7999]**. Free-form changes-0.1595 and Yes/No+0.1280, both non-significant. The only clear subgroup difference is `where`-5.3790 for multi-layer (23/45, `p=0.01034`).
- Diagnostics versus the retained Layer17 reference show two distinct non-winning regimes. Layer18 has slot cosine0.6681, visual-attention entropy0.7465, Cross-Attention delta/query0.8842 and text delta/anchor0.4393, weaker than Layer17's0.7118/0.7133/1.1751/0.4987. Multi-layer raises Cross delta/query to1.2015 and text delta/anchor to0.5427 while making visual attention more selective at0.6028, but slot cosine rises to0.7611 and accuracy does not recover. Therefore merely strengthening or repeating the pathway does not guarantee useful evidence; the layer17 representation is better aligned with open/spatial answer generation under this fixed architecture.
- Conclusion: retain **Layer17-only** as the final anchor. Moving toLayer18 significantly harms Overall and Free-form, while shared Layer17+18+19 adds no Overall benefit over Layer18 and sharply worsens spatial `where` accuracy. This establishes an empirical layer-sensitivity result, not a universal claim that layer17 is theoretically unique. Stop the layer search; no Layer16/19/20 or additional layer combinations are justified within the frozen protocol.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l18_p20_s8_av10_seed44_20260903` and `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_18_19_shared_p20_s8_av10_seed44_20260903`; selected checkpoints `checkpoints/epoch_3`, summaries and predictions under `eval_validation/epoch_3`.

### 2026-09-02 - pathvqa_qdpt_d768_question_q10_l17_p20_no_static_visual_seed44

- Commit/config: commits `20d8ebb` and checkpoint-reload fix `6e1e745`; D768 QDPT seed44 with the entire layer17 static visual insertion removed. The controlled model has no private `S_v8`, no static visual `A_v10`, no visual token insertion/strip and no dynamic visual write. It preserves P20, text anchor A_t10, question-attention-pooled Q10, full pre-Block17 visual K/V, CA768/16, Z10 and `Z -> dynamic LLM Prompt` exactly.
- Dataset and split: PathVQA official Validation, all6,259 questions and832 image clusters; fixed epoch3-only full evaluation. Seed/data seed44/42.
- Controlled change versus D768 seed44 reference: remove all18 Layer17 Static Visual Prompt Tokens (`S_v8+A_v10`) and execute the frozen Block17 once on its unmodified original visual sequence. Directional CA continues to read the same full pre-Block17 visual Token source.
- Trainable parameters: **7,786,752** = soft/text anchors76,800 + Directional/Text modules7,709,952; sparse visual group0. This removes only18,432 parameters from the7,805,184 reference. Training runtime6,142.33s,1,845 steps and train loss11.39386; Prompt LR0.3 and Directional LR1e-4 unchanged.
- Validation Overall: **58.4438**, standalone image-clustered95% CI **[57.0164,59.8655]**. Yes/No / Free-form: **90.9440 /26.0370**. Per question type: how10.8527, other7.1429, what21.7425, when0.0000, where60.1467, why4.7619, yes/no90.9440; Free-form type macro17.4411.
- Paired against the exact D768 seed44 reference59.5622: **-1.1184 Overall**; no-static-only/reference-only correct261/331, exact McNemar `p=0.004530`, image-clustered paired95% CI **[-1.8938,-0.3484]**. Yes/No changes only-0.0960, while Free-form falls **-2.1378**. The largest type loss is `where` **-12.4694**, followed by how-1.5504 and what-0.5495.
- Diagnostics remain active rather than collapsed: slot cosine0.6407, question-pooling entropy0.6430, visual-attention entropy0.6484, Cross-Attention delta/query1.2556, text delta/anchor0.5019 and Z norm33.8075. Static visual write, visual Prompt norms and sparse visual gradients are exactly zero by construction. The text-conditioned route is therefore healthy but cannot recover the missing visual calibration effect.
- Conclusion: reject deletion of the complete static visual insertion. Its gain is statistically significant and concentrated in open/spatial questions, while binary accuracy is unchanged. The former `S_v8` and `A_v10` have no functional distinction after dynamic visual write removal beyond slot position, optimizer group and learning rate; if retained in the paper they should be presented jointly as **18 Layer17 Static Visual Prompt Tokens**, not as two semantic modules. This experiment establishes necessity of the combined static visual Prompt set, not the individual contribution of8 versus10 tokens.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_no_static_visual_seed44_20260902`; checkpoint `checkpoints/epoch_3`, Validation under `eval_validation/epoch_3`, diagnostics `dynamic_prompt_diagnostics.jsonl`, report `train_report.json`. Paired analysis was generated in server-temporary `/tmp/qdpt_no_static_compare_20260902` and can be reproduced from the two saved prediction files.

### 2026-09-02 - PathVQA QDPT-D768 training peak VRAM

- Method/config: locked PathVQA Directional Text-Dynamic-Only D768 training configuration used by the Day 2 multi-seed suite; observation was made during the QDPT-D768 phase.
- GPU: GPU 0, single-GPU training.
- Observed peak device-memory usage: **25,349 MiB (24.75 GiB)**.
- Observation timestamp: **2026-09-02 10:58:46 +08:00**.
- Measurement source: user-provided AutoDL monitoring-panel screenshot. Treat this as an external device-level peak observation, which may include CUDA context and non-PyTorch allocations; it is not interchangeable with `torch.cuda.max_memory_allocated()`.
- Comparison status: pending a Full-Attention LoRA-r8 measurement collected through the same monitoring interface.

### 2026-09-04 - PathVQA Full-Attention LoRA-r8 multi-seed replication (seeds45-46)

- Exact experiments: `pathvqa_lora_full_model_attention_r8_seed45_20260904` and `pathvqa_lora_full_model_attention_r8_seed46_20260904`; commit `f3dc3c3`. Both use Full-Attention LoRA-r8 with alpha16, dropout0.05 and LR1e-4 on all24 visual Attention `qkv/proj` modules and all36 LLM Self-Attention `q/k/v/o` modules. Dataset/data seed remain PathVQA train19,654 /42; each run trains3 epochs with micro-batch1 and gradient accumulation32, then evaluates only epoch3 on the complete Validation split of6,259 questions and832 image clusters. Test was not run.
- Seed45: Overall **59.2427**, standalone image-clustered95% CI **[57.7052,60.6885]**; Yes/No / Free-form **91.4560 /27.1219**. Per question type: how11.6279, other14.2857, what21.1538, when7.6923, where71.3936, why4.7619 and yes/no91.4560; Free-form question-type macro21.8192. Training status passed with1,845 steps, runtime14,766.43s and train loss0.585167. Evaluation TTFT mean0.057278s, weighted TPOT0.029367s/token and34.052 decode tokens/s.
- Seed46: Overall **59.2107**, standalone image-clustered95% CI **[57.6666,60.6539]**; Yes/No / Free-form **91.5200 /26.9943**. Per question type: how12.4031, other21.4286, what20.6044, when7.6923, where73.3496, why4.7619 and yes/no91.5200; Free-form question-type macro23.3733. Training status passed with1,845 steps, runtime14,931.07s and train loss0.585241. Evaluation TTFT mean0.057887s, weighted TPOT0.030856s/token and32.408 decode tokens/s.
- Parameter and target audit: both runs report exactly **7,077,888** trainable parameters, matching the prediction, and select all expected48 visual plus144 language Attention Linear modules. Both output directories contain the final adapter, training report, epoch3 Validation summary, comparisons and predictions; neither run has a partial-evaluation or completion failure.
- Three-seed LoRA summary (44/45/46): Overall **59.2640 +/- 0.0666** sample standard deviation; Yes/No **91.7227 +/- 0.4077**; Free-form **26.8986 +/- 0.2836**. The corresponding locked QDPT-D768 means are59.0030+/-0.4859,91.2107+/-0.9713 and26.8879+/-1.4185, so QDPT minus LoRA is **-0.2610 Overall, -0.5120 Yes/No and -0.0107 Free-form** on the three-seed means. Matched Overall differences for seeds44/45/46 are+0.2236/-0.5592/-0.4473; the direction is not consistently favorable to QDPT.
- Efficiency comparison: QDPT-D768 uses7,805,184 parameters, **10.27% more** than LoRA-r8, but its seed45/46 training runtimes6,473.07/6,439.31s are about2.29x faster than LoRA's14,766.43/14,931.07s under these recorded runs. Device-level LoRA peak VRAM remains unrecorded and must not be inferred from runtime or parameter count.
- Conclusion: the defensible PathVQA claim is performance parity, not superiority. LoRA has a0.261-point mean Overall advantage and much lower seed variance; QDPT has essentially identical mean Free-form performance, a transparent question-to-visual-evidence path and substantially shorter observed training time. QDPT's seed44 Free-form advantage does not reproduce as a stable mean advantage, so the paper must not present that single-seed decomposition as the general result. Formal pooled or per-seed paired significance analysis remains separate from this descriptive three-seed summary.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r8_seed45_20260904` and `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r8_seed46_20260904`; selected checkpoints `checkpoints/epoch_3`, summaries and predictions under `eval_validation/epoch_3`, reports `train_report.json`.

### 2026-09-04 - PathVQA QDPT-D768 learned-static-query control seed44

- Exact experiment: `pathvqa_qdpt_d768_learned_q10_l17_p20_s8_av10_seed44_20260904`; PathVQA official train19,654 and complete Validation6,259 with832 image clusters, seed/data seed44/42,3 epochs and epoch3-only full evaluation. Test was not run.
- Controlled change: replace the final QDPT-D768 question-attention-pooled `Q10` with ten directly learned static Query tokens. The retained `S8@3e-5+A_v10@1e-4` Layer17 static visual Prompt, full Layer17 visual K/V, D768/16-head Directional Cross-Attention, Z10, P20/A_t10 text Prompt interface, optimizer, training schedule and **7,805,184** trainable-parameter count remain fixed.
- Validation Overall: **57.1817**, standalone image-clustered95% CI **[55.7038,58.6157]**; Yes/No / Free-form **89.6000 /24.8564**. Per question type: how10.8527, other7.1429, what20.2904, when0.0000, where60.3912, why0.0000 and yes/no89.6000; Free-form question-type macro16.4462.
- Paired against the exact question-guided D768 seed44 reference59.5622: **-2.3806 Overall**, static-only/question-guided-only correct205/354, exact McNemar `p=3.05e-10`, image-clustered paired95% CI **[-3.1377,-1.6077]** with10,000 bootstrap iterations. Yes/No falls1.4400 and Free-form3.3184; the largest question-type losses are `where` -12.2249, `why` -4.7619, `what` -2.0016 and `how` -1.5504.
- Training and diagnostics:1,845 steps, runtime6,324.05s and train loss11.42890. The static-query path is active rather than dead: final shared Workspace gradient norm is0.99975 and Cross-Attention delta/query reaches4.1466. However, slot pairwise cosine rises to0.9336 near training end; full-validation means remain0.6713 slot cosine,0.7596 visual-attention entropy,2.1994 Cross delta/query and0.3013 text delta/anchor. This indicates that unconditioned learned queries can strongly read the image but tend toward generic, redundant visual summaries.
- Conclusion: the current-question-conditioned Query is a necessary part of QDPT, not a replaceable parameter-matched learned-query/Q-Former-style aggregator. The statistically significant loss is especially concentrated in open and spatial answers, matching the intended role of question-guided evidence selection. This is a completed mechanism ablation; do not spend additional seeds on the learned-static-query control.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_learned_q10_l17_p20_s8_av10_seed44_20260904`; selected checkpoint `checkpoints/epoch_3`, Validation summary and predictions under `eval_validation/epoch_3`, diagnostics `dynamic_prompt_diagnostics.jsonl`, report `train_report.json`.

### 2026-09-04 - SLAKE final QDPT-D768 multi-seed evaluation (seeds44-46)

- Exact experiments: `slake_qdpt_d768_question_q10_l17_p20_s8_av10_seed44_20260904`, `slake_qdpt_d768_question_q10_l17_p20_s8_av10_seed45_20260904` and `slake_qdpt_d768_question_q10_l17_p20_s8_av10_seed46_20260904`; final frozen QDPT-D768 configuration at commit `b6736f1`. Each run uses `S8@3e-5+A_v10@1e-4`, question-attention-pooled Q10, full Layer17 visual K/V, D768/16-head Directional Cross-Attention, Z10 written only to the LLM Prompt, P20/A_t10 and no dynamic visual-Z write. Each has exactly **7,805,184** trainable parameters.
- Dataset and protocol: SLAKE train split for3 epochs, seed/data seeds44-46/42; only epoch3 is evaluated on the complete official Test split of2,094 questions. All three summaries report zero missing and zero extra predictions. This is direct architecture and optimizer transfer from PathVQA; no SLAKE-specific width, layer, Prompt-length or learning-rate tuning was performed.
- Seed44: Overall **77.65**; CLOSED / OPEN **84.45 /73.13**; KVQA / VQA **64.04 /79.64**; EN / ZH **77.95 /77.35**. Training runtime4,610.30s, train loss3.29032. Evaluation TTFT mean0.064015s, weighted TPOT0.015924s/token and62.800 decode tokens/s.
- Seed45: Overall **76.70**; CLOSED / OPEN **83.01 /72.50**; KVQA / VQA **59.18 /79.26**; EN / ZH **77.47 /75.90**. Training runtime4,605.77s, train loss3.43672. Evaluation TTFT mean0.063552s, weighted TPOT0.015548s/token and64.316 decode tokens/s.
- Seed46: Overall **76.74**; CLOSED / OPEN **83.13 /72.50**; KVQA / VQA **60.67 /79.09**; EN / ZH **77.47 /75.99**. Training runtime4,617.10s, train loss3.58733. Evaluation TTFT mean0.064380s, weighted TPOT0.016588s/token and60.286 decode tokens/s.
- Three-seed summary: Overall **77.03 +/- 0.54** sample standard deviation; CLOSED **83.53 +/- 0.80**; OPEN **72.71 +/- 0.36**; KVQA **61.30 +/- 2.49**; VQA **79.33 +/- 0.28**; EN **77.63 +/- 0.28**; ZH **76.41 +/- 0.81**. Mean training runtime is4,611.06s.
- Diagnostics: all three runs retain active question-conditioned visual reading. Full-Test visual-attention entropy is0.9112/0.8603/0.9448 and Cross-Attention delta/query is2.2561/1.7190/1.8844 for seeds44/45/46. Slot cosine is high at0.9540/0.8844/0.8967, but the complete path remains finite and produces stable Overall and OPEN scores. The largest seed variation is concentrated in KVQA rather than VQA or OPEN.
- Conclusion: the frozen QDPT-D768 method transfers to SLAKE at approximately77.0 Overall with moderate Overall variance and stable OPEN/VQA behavior. Seed44 exceeds the historical Static Prompt seed44 result74.40 by3.25 points while using more parameters; relative claims against Full-Attention LoRA must wait for the matched SLAKE LoRA run. The prior high-capacity Full Workspace seed44 remains0.72 points higher at78.37 but uses76.90M parameters, so D768 preserves most of that accuracy with roughly one tenth the trainable parameters.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_qdpt_d768_question_q10_l17_p20_s8_av10_seed44_20260904`, `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_qdpt_d768_question_q10_l17_p20_s8_av10_seed45_20260904` and `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_qdpt_d768_question_q10_l17_p20_s8_av10_seed46_20260904`; selected checkpoints `checkpoints/epoch_3`, summaries and predictions under `eval_test/epoch_3`, reports `train_report.json`.
- Operational correction (2026-09-05): `slake_qdpt_d768_question_q10_l17_p20_s8_av10_seed44_20260905` was an accidental exact duplicate started after the completed directories were overlooked. The user manually interrupted it within minutes; it has no completed checkpoint or evaluation and is not a valid result. Do not resume or include it in any table.

### 2026-09-05 - SLAKE Full-Attention LoRA-r8 multi-seed evaluation (seeds44-46)

- Exact experiments: `slake_lora_full_model_attention_r8_seed44_20260905`, `slake_lora_full_model_attention_r8_seed45_20260905` and `slake_lora_full_model_attention_r8_seed46_20260905`; commit `7b1c16a`. All runs apply rank8, alpha16, dropout0.05 LoRA with LR1e-4 to all24 visual Attention `qkv/proj` modules and all36 LLM Self-Attention `q/k/v/o` modules. The audit selects exactly48 visual plus144 language Linear targets and reports exactly **7,077,888** trainable parameters.
- Dataset and protocol: SLAKE train split9,834 samples,3 epochs, seed/data seeds44-46/42, micro-batch1 and gradient accumulation32. Only the epoch3 adapter is evaluated on all2,094 official Test questions with `answer_mode=raw`; every run reports zero missing and zero extra predictions.
- Seed44: Overall **81.95**; CLOSED / OPEN **86.84 /78.70**; KVQA / VQA **71.91 /83.42**; EN / ZH **83.13 /80.74**. Training runtime7,028.57s and train loss0.217572. TTFT mean0.071506s, weighted TPOT0.023910s/token and41.824 decode tokens/s.
- Seed45: Overall **81.57**; CLOSED / OPEN **87.44 /77.66**; KVQA / VQA **68.54 /83.47**; EN / ZH **82.38 /80.74**. Training runtime7,068.04s and train loss0.217225. TTFT mean0.071921s, weighted TPOT0.024151s/token and41.405 decode tokens/s.
- Seed46: Overall **81.95**; CLOSED / OPEN **86.72 /78.78**; KVQA / VQA **70.79 /83.58**; EN / ZH **83.03 /80.83**. Training runtime7,020.87s and train loss0.217709. TTFT mean0.072089s, weighted TPOT0.024591s/token and40.665 decode tokens/s.
- Three-seed summary: Overall **81.82 +/- 0.22** sample standard deviation; CLOSED **87.00 +/- 0.39**; OPEN **78.38 +/- 0.62**; KVQA **70.41 +/- 1.72**; VQA **83.49 +/- 0.08**; EN **82.85 +/- 0.41**; ZH **80.77 +/- 0.05**. Mean training runtime is7,039.16s.
- Matched comparison against QDPT-D768: QDPT minus LoRA Overall is **-4.30/-4.87/-5.21** for seeds44/45/46, and the three-seed mean gap is **-4.79**. QDPT-only/LoRA-only correct counts are56/146,53/155 and53/162, with exact McNemar `p=1.91e-10`, `8.62e-13` and `4.94e-14`. The gap is therefore large, directionally identical and paired-significant in every seed, not seed variance or an evaluation-set mismatch.
- Breakdown comparison of three-seed means, QDPT minus LoRA: CLOSED **-3.47**, OPEN **-5.67**, KVQA **-9.11**, VQA **-4.16**, EN **-5.22** and ZH **-4.36**. The largest deficit is KVQA and the deficit is larger on OPEN than CLOSED, indicating that QDPT's frozen-LLM Prompt interface is especially limited for domain knowledge/answer mapping and open generation; this is not merely binary-answer calibration.
- Efficiency tradeoff: QDPT-D768 uses7,805,184 parameters,10.27% more than LoRA-r8, but its mean training runtime4,611.06s is1.53x faster than LoRA. QDPT mean TTFT is0.063982s and mean weighted TPOT0.016020s/token, versus LoRA0.071839s and0.024217s/token; the Prompt method leaves frozen-LLM decoding substantially lighter despite lower task accuracy. Runtime observations are implementation-specific and must not be presented as hardware-independent complexity results.
- Conclusion: unlike PathVQA, SLAKE decisively rejects performance parity between final QDPT-D768 and Full-Attention LoRA-r8. Even the76.90M-parameter Full Workspace seed44 result78.37 remains3.58 points below matched LoRA seed44, so simply restoring capacity is not a credible fix. The defensible cross-dataset story is a clear accuracy-efficiency tradeoff and a mechanism-focused Prompt method, not universal LoRA replacement or parameter-performance superiority.
- Publication placement decision (2026-09-06): preserve this complete result as a different-adaptation-family reference in the appendix rather than repeatedly centering the main paper on LoRA. The controlled main table will instead compare frozen-backbone Prompt methods under the same protocol. The main text must still acknowledge once that the appendix reference is stronger on SLAKE; appendix placement changes narrative scope, not the factual conclusion or reporting obligation.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/lora/slake_lora_full_model_attention_r8_seed44_20260905`, `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/lora/slake_lora_full_model_attention_r8_seed45_20260905` and `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/lora/slake_lora_full_model_attention_r8_seed46_20260905`; selected adapters `checkpoints/epoch_3`, summaries and predictions under `eval_test/epoch_3`, reports `train_report.json`.

### 2026-09-07 - PathVQA QDPT-D768 question-only / w/o visual CA seed44

- Exact experiment: `pathvqa_qdpt_d768_question_only_q10_l17_p20_s8_av10_seed44`; PathVQA official train19,654 and complete Validation6,259 with832 image clusters, seed/data seed44/42,3 epochs and epoch3-only full evaluation. Test was not run.
- Controlled change: preserve the retained `S8@3e-5+A_v10@1e-4` Layer17 static visual Prompt, question-attention-pooled Q10, P20/A_t10 text Prompt interface, D768 output transformation, optimizer, data order and initialization, but replace `Z=Q+CA(Q,V,V)` with `Z=Q`. The original image and question still enter the frozen Qwen3-VL normally; only the auxiliary Directional visual K/V read is disabled. The Cross-Attention parameters remain instantiated for bit-identical initialization and checkpoint structure but are dormant and receive no gradient.
- Parameter and training audit: nominal checkpoint trainable structure remains **7,805,184** parameters, including the dormant visual-CA tensors; do not interpret that number as the active parameter requirement of the question-only path. Training completed1,845 steps in6,450.67s with train loss11.48642. Evaluation audit reports `visual_conditioning=question_only`, zero visual tokens/projection/Cross-Attention delta and no visual dynamic write.
- Validation Overall: **57.6290**, standalone image-clustered95% CI **[56.1769,58.9892]**. Yes/No / Free-form: **89.7280 /25.6222**. Per question type: how10.0775, other14.2857, what20.8006, when0.0000, where62.8362, why4.7619 and yes/no89.7280; Free-form question-type macro18.7936.
- Paired against the exact full QDPT-D768 seed44 reference59.5622: **-1.9332 Overall**; question-only/full-QDPT exclusive correct counts237/358, exact McNemar `p=7.98e-7`, image-clustered paired95% CI **[-2.7582,-1.1009]** with10,000 bootstrap iterations. Yes/No falls **-1.3120** (87/128, `p=0.00624`) and Free-form falls **-2.5526** (150/230, `p=4.77e-5`). The largest type loss is `where` **-9.7800** (18/58, `p=4.71e-6`), followed by how-2.3256 and what-1.4914.
- Diagnostics: the remaining question-conditioned text route is active and finite rather than collapsed. Full-Validation means are question/workspace norm25.8395, text delta/anchor0.5402, text-pooling entropy0.6280 and slot pairwise cosine0.6132; all visual-conditioning and Cross-Attention diagnostics are exactly zero by construction. TTFT mean is0.050834s, weighted TPOT0.019472s/token and decode speed51.356 tokens/s.
- Conclusion: question-conditioned Prompt generation alone is useful but insufficient. Correct visual K/V reading contributes a paired-significant1.93-point gain affecting both binary and open answers, with the strongest effect on spatial questions. Together with the previous visual-K/V mismatch loss of7.03 points, the mechanism evidence is now complete: removing visual evidence hurts, replacing it with incorrect evidence hurts much more, and correctly aligned question-to-visual Cross-Attention performs best. Retain the visual-CA path and mark the question-only control complete; no additional seed is justified for this mechanism ablation.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_only_q10_l17_p20_s8_av10_seed44_20260907`; selected checkpoint `checkpoints/epoch_3`, Validation summary and predictions under `eval_validation/epoch_3`, diagnostics `dynamic_prompt_diagnostics.jsonl`, report `train_report.json`.

### 2026-09-07 - Electrical QDPT-D768 startup failure

- Exact experiment: `electrical_qdpt_d768_question_q10_l17_p20_s8_av10_seed44`; requested through `RUN_TARGET=electrical_qdpt_d768_seed44` on the private electrical dataset, seed44. No training step or evaluation was reached, so no score, checkpoint or parameter-performance conclusion exists.
- Intended controlled configuration: final QDPT-D768 with question-attention-pooled Q10, Layer17 visual K/V, retained `S8+A_v10` static visual Prompt, P20/A_t10 and no dynamic visual write. The launcher CPU suite passed44 tests before model loading.
- Failure: dataset construction passed `TRAIN_EXPERT_JSONS`, intentionally defined as a tuple of eight JSON paths, into `FourViewMMRLDataset`. Both `normalize_json_paths()` and the subsequent `load_jsons()` recognized only `list` as a multi-file collection. `normalize_json_paths()` therefore called `os.fspath()` on the tuple and raised `TypeError: expected str, bytes or os.PathLike object, not tuple` before the first batch.
- Resolution: treat both list and tuple inputs as multi-file path collections in `normalize_json_paths()` and `load_jsons()`, with a regression test that loads two tuple-provided JSON sources and preserves per-item source paths. This is an input-container compatibility bug, not a failed model hypothesis. Rerun the same target after pulling the fix; do not change experiment naming or hyperparameters.
- Correction after the second startup attempt: the first fix exposed the analogous image-root bug one stage later. `TRAIN_EXPERT_IMAGE_DIRS` is a tuple of four roots, while `build_image_mapping()` also recognized only lists; it returned the tuple as a supposed single root and `_resolve_img()` failed at `os.path.join(tuple, image_file)` before dataset sampling. No training step, checkpoint or score was produced. The complete resolution extends tuple support to `build_image_mapping()` and adds a two-root mapping regression test. Before another GPU launch, the real eight-JSON/four-root configuration must pass a server-side no-model path-resolution preflight.
### 2026-09-08 - pathvqa_grasp_reimpl_n4_h512_seed44_20260907

- Commit/config: initial independent GRASP approximation at `02a7aca`; `N=4`, bottleneck `h=512`, Entmax-1.5, four trainable region Prompt prototypes, frozen Qwen3-VL, AdamW LR1e-4, weight decay0.01,10% warmup and linear decay.
- Dataset and split: PathVQA train19,654; complete official Validation6,259 questions with832 image clusters. Only epoch3 was evaluated; Test was not run.
- Seed / data seed:44 /42.
- Controlled method: post-`visual.merger` tokens are pooled into four fixed spatial blocks; a frozen-LLM context representation and projected block features route four Prompt prototypes into one global Prompt token. Trainable parameters are exactly **2,632,704** =10,240 Prompt-prototype parameters +2,622,464 projection parameters.
- Validation Overall: **45.0871**; image-clustered95% CI **[43.8178,46.2854]**.
- Yes/No / Free-form: **82.9120 /7.3708**. Per question type: how1.5504, other0.0000, what6.0440, when0.0000, where18.3374, why0.0000 and yes/no82.9120; Free-form question-type macro4.3220.
- Training:3 epochs and1,845 steps; runtime6,660.85s, reported train loss23.5023. Logged epoch-window loss means fall from38.447 in epoch1 to16.826 in epoch2 and15.784 in epoch3; the final logged loss is15.36. Raw global gradient norms remain extremely large (epoch means331.1/222.6/249.0) and are clipped to1.0.
- Diagnostics: all93 records are finite. Across all records, Entmax zero-weight fraction averages0.00134, normalized routing entropy0.9645 and maximum region weight0.3348. Over the last30 records the zero fraction is exactly0, entropy0.9798 and maximum weight0.3190, so the intended sparse region selection converges toward near-uniform four-block averaging. Prompt-prototype norm changes only from approximately2.03 to2.08; final-window global Prompt norm is0.5822. Prototype and projection gradients remain active at0.9852 and0.1636, respectively, ruling out a detached branch.
- Implementation audit after completion: this run is **not accepted as a faithful GRASP baseline**. The global Prompt was reserved at the absolute start of the full chat sequence instead of being concatenated immediately with the visual-token segment as required by Eq.6-7. The frozen-LLM query encoded all pre-answer nonvisual chat/template tokens rather than the raw question tokens specified by Eq.2. The unified3-epoch protocol also differs materially from the paper's up-to20 epochs with patience5; at LR1e-4 the loss is still descending and the Prompt barely moves. These implementation/protocol deviations precede any claim that GRASP itself is unsuitable for medical VQA.
- Conclusion: the45.09 result is a valid negative record for the initial approximation but must not enter the direct-comparison main table. Correct the Prompt placement and exact question-only encoding first; then decide explicitly whether to retain the common3-epoch budget or grant the reproduced baseline its published convergence protocol. Do not launch SLAKE from the flawed checkpoint path. The shell history contains `RUN_TARGET=slake_grasp_seed44`, but no SLAKE GRASP output directory, process, summary or log exists on the current container, so no SLAKE experiment is recorded as completed.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/grasp/pathvqa_grasp_reimpl_n4_h512_seed44_20260907`; checkpoint `checkpoints/epoch_3`; Validation summary/predictions under `eval_validation/epoch_3`; diagnostics `grasp_diagnostics.jsonl`; training report `train_report.json`.
### 2026-09-08 - PathVQA QDPT-D768 10-epoch convergence marathon seed44

- Exact experiment: `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_marathon10_seed44`; commit `feb4fb9`. Final QDPT-D768 architecture with P20, text anchor A_t10, question-guided Q10, full code-index17 visual K/V, D768/16-head Cross-Attention, static visual `S8@3e-5+A_v10@1e-4`, no dynamic visual write and exactly **7,805,184** trainable parameters.
- Dataset/protocol: PathVQA official train19,654, model/data seeds44/42, effective batch32 and10 epochs. The linear scheduler horizon is10 epochs, so its epoch3 is not a bitwise reproduction of the former3-epoch scheduler endpoint. Complete Validation6,259 questions /832 image clusters was evaluated after every epoch3-10; Test was not run.
- Epoch3-10 Overall: **56.4467 /57.2456 /58.6675 /58.7794 /57.4852 /57.1817 /57.3894 /57.1657**. Corresponding Yes/No:89.9840/89.6000/90.5920/91.2320/90.3680/90.7520/90.4000/90.7200; Free-form:23.0057/24.9840/26.8347/26.4199/24.6969/23.7077/24.4735/23.7077.
- Best marathon checkpoint: **epoch6**, Overall **58.7794**, Yes/No **91.2320**, Free-form **26.4199**, standalone image-clustered95% CI **[57.2920,60.2058]**. Per question type: how11.6279, other14.2857, what20.8791, when7.6923, where67.2372, why14.2857 and yes/no91.2320; Free-form type macro22.6680.
- Comparison with the retained3-epoch seed44 result59.5622: marathon best is **-0.7828 Overall**, +0.1920 Yes/No and -1.7550 Free-form. Epoch10 is **-2.3965 Overall** below the retained result. Because scheduler horizons differ, this is a protocol comparison rather than a paired checkpoint continuation experiment.
- Optimization/runtime:6,150 optimizer steps, total reported runtime27,712.15s (7.70h, including in-process Validation callbacks) and average train loss8.79945. Latest logged loss across epoch3-10 is9.6446/9.0091/8.6603/8.0499/8.7339/7.1639/5.8936/5.3965. Validation peaks at epoch6 while training fit continues improving, providing direct evidence that later optimization does not improve generalization under this schedule. The eight full evaluations consumed approximately94.95 minutes in total.
- Best-checkpoint timing: TTFT mean0.052259s, weighted TPOT0.019584s/token and51.062 decode tokens/s. All6259 requests completed successfully.
- Conclusion: the10-epoch budget does not rescue or improve QDPT. Retain the original3-epoch linear schedule and its multi-seed results; do not launch additional QDPT long-training or learning-rate-search runs during paper completion. The negative result informs convergence reporting but does not replace the main checkpoint.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_marathon10_seed44_20260908`; checkpoints`epoch_3` through`epoch_10`, per-epoch summaries under`eval_validation/epoch_N`, compact record`marathon_progress.tsv`, report`train_report.json`. Figure: `figures/qdpt_d768_marathon_curve.svg` and PNG counterpart.

### 2026-09-09 - PathVQA corrected-query GRASP with reversed Prompt order

- Exact experiment: `pathvqa_grasp_reimpl_corrected_n4_h512_seed44_20260909`; commits `56e84ed` and `6a4bdd7`. PathVQA train19,654, complete Validation6,259 /832 image clusters, seed/data seed44/42,3 epochs and epoch3-only evaluation. The run completed training and evaluation with the intended2,632,704 trainable parameters.
- Controlled changes relative to the initial45.0871 approximation: encode separately tokenized raw-question tokens with the frozen LLM instead of pooling the pre-answer chat context, and move the single global Prompt from the absolute chat start to the position immediately after the visual segment. N=4, h=512, Entmax-1.5, optimizer and training protocol remained fixed.
- Validation Overall: **40.7254**; image-clustered95% CI **[39.37,41.97]**. Yes/No / Free-form: **74.91 /6.64**. Relative to the initial approximation, Overall changes **-4.3617**, Yes/No approximately-8.00 and Free-form approximately-0.73. Relative to Frozen Base34.77 it remains about+5.96 Overall, but it is far below the PathVQA Static Prompt epoch3 Validation54.87.
- Evaluation timing visible in the completed summary: TTFT mean0.066767s and TPOT0.019155s/token (52.206tokens/s). All6,259 predictions completed.
- Post-run paper audit found a decisive implementation error: GRASP Eq.(6) defines `T'=[p_global,T]`, and Eq.(7) defines `[T',X_Q]=[p_global,T,X_Q]`. This run instead implemented `[T,p_global,X_Q]`. In a causal decoder the visual tokens cannot attend to a Prompt placed after them, so this is not a faithful test of the published ordering.
- Conclusion: preserve40.7254 as a completed negative engineering record, but exclude it from every GRASP comparison table and do not interpret it as architecture failure. The raw-question correction is faithful; the Prompt order is reversed. The only justified rerun is a position-only correction that inserts the Prompt immediately before Qwen's visual segment while preserving the raw-question path and all hyperparameters.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/grasp/pathvqa_grasp_reimpl_corrected_n4_h512_seed44_20260909`; checkpoint`checkpoints/epoch_3`, summary/predictions under`eval_validation/epoch_3`, diagnostics`grasp_diagnostics.jsonl`, report`train_report.json`.

### 2026-09-09 - PathVQA paper-order GRASP reimplementation seed44

- Exact experiment: `pathvqa_grasp_reimpl_paper_order_n4_h512_seed44`; commit `6342f0d`. PathVQA train19,654 and complete Validation6,259 /832 image clusters, seed/data seed44/42,3 epochs and epoch3-only evaluation. The checkpoint contains exactly **2,632,704** trainable parameters:10,240 Prompt-prototype and2,622,464 projection parameters.
- Controlled change: relative to the preceding40.7254 engineering run, move the single generated global Prompt from after the visual segment to immediately before Qwen's `<|vision_start|>`, yielding the paper-specified causal order `[Prompt, Visual, Question]`. The separately tokenized raw-question frozen-LLM encoding, post-merger4-block visual grid, `N=4`, `h=512`, Entmax-1.5, optimizer and all data/training settings remain fixed.
- Validation Overall: **39.7508**; image-clustered95% CI **[38.5286,40.9129]**. Yes/No / Free-form: **75.3600 /4.2438**. Per question type: how1.5504, other0.0000, what2.7865, when0.0000, where14.6699, why0.0000 and yes/no75.3600; Free-form question-type macro3.1678.
- Relative to the incorrect post-visual Prompt run, paper-order placement changes Overall by **-0.9746**, Yes/No by+0.45 and Free-form by-2.3962. Relative to Frozen Base34.77 it is approximately+4.98 Overall, but remains far below Static Prompt and QDPT under the same PathVQA evaluation family.
- Diagnostics rule out a dead router. Late Entmax zero-weight fraction varies0.125-0.500, normalized entropy0.35-0.52 and maximum region weight0.58-0.80; Prompt prototypes receive gradient norm about0.99 and projection gradients remain nonzero. The generated global Prompt norm remains only0.76-0.88 while question and visual-block norms are roughly130-161 and40-44. The branch learns sparse spatial selection, but its single prototype-mixture Token does not carry enough transferable answer evidence for PathVQA, especially Free-form generation.
- Evaluation timing: TTFT mean0.062563s, weighted TPOT0.015733s/token,63.56 decode tokens/s and37.201 end-to-end generated tokens/s; all6,259 requests completed.
- Conclusion: this is the first formally usable GRASP reimplementation result after correcting both question input and causal Prompt order. It establishes that GRASP transfers poorly under the locked3-epoch Qwen3-VL/PathVQA protocol; it does not establish that the original20-epoch remote-sensing method is universally ineffective. Stop GRASP hyperparameter and SLAKE runs in the current project. If reported, label it clearly as an independent reimplementation under the unified protocol rather than an official score.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/grasp/pathvqa_grasp_reimpl_paper_order_n4_h512_seed44_20260909`; checkpoint`checkpoints/epoch_3`, summary/predictions under`eval_validation/epoch_3`, diagnostics`grasp_diagnostics.jsonl`, report`train_report.json`.

### 2026-09-09 - PathVQA CoCoOp-style P20/H160 seed44

- Exact experiment: `pathvqa_cocoop_style_p20_h160_seed44_20260909`; PathVQA train/complete official Validation, seed/data seed44/42,3 epochs and epoch3-only full evaluation. Test was not run.
- Controlled method: frozen post-merger LLM-space visual Tokens are mean pooled and passed through a normally initialized `2560->160->2560` Meta-Net. Its single image-conditioned bias is added to all20 embedding-row-initialized static Prompt Tokens before the complete chat. The method has no question-Token access, Layer17 Prompt, visual-encoder write or QDPT Directional CA. Prompt/Meta-Net learning rates are0.3/3e-4.
- Trainable parameters: **873,120** exactly =51,200 static P20 parameters +821,920 Meta-Net parameters.
- Validation Overall: **57.4053** (3,593/6,259; displayed57.41); image-clustered95% CI **[55.9473,58.7765]** across832 image clusters. Yes/No / Free-form: **89.8560 /25.0479**. Per question type: how9.3023, other14.2857, what20.9576, when0.0000, where57.7017, why4.7619 and yes/no89.8560; Free-form question-type macro17.8349.
- Relative point estimates: versus PathVQA Static Prompt epoch3 Validation54.87, CoCoOp-style gains approximately+2.54 Overall while using an image-conditioned Meta-Net. Versus matched-seed QDPT-D76859.5622 it is-2.1569, and versus the Sandwich placement60.7765 it is-3.3712. The largest mechanism-relevant deficit is `where`:57.7017 versus72.6161 for QDPT and76.0391 for Sandwich, supporting question-guided visual evidence selection beyond global image conditioning. Paired significance is not yet computed.
- Training completed3 epochs in5,962.33s with train loss11.74265. The Meta-Net is active rather than suppressed: over the final eight diagnostic windows its gradient norm remains0.9989-0.9992, while soft-Prompt gradient norm is0.0407-0.0463. Image-feature norm is20.97-24.84, generated bias norm43.28-54.85 and bias/static Prompt norm ratio0.180-0.229. The result therefore reflects an optimized image-conditioned branch, not detachment or collapse.
- Evaluation timing: TTFT mean0.048588s, weighted TPOT0.019554s/token,51.14 decode tokens/s and request mean0.099745s.
- Conclusion: the CoCoOp-style baseline is useful and parameter-efficient rather than collapsed: image-only conditional Prompting materially improves the static baseline with only0.873M trainable parameters. It remains clearly below question-guided QDPT, especially on spatial questions, so global image conditioning does not replace current-question Q to visual K/V retrieval. Do not enlarge H160 merely to chase QDPT until diagnostics are recovered; retain this as the faithful lightweight primary CoCoOp-style result and treat any future capacity-matched variant as a separate control.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/cocoop/pathvqa_cocoop_style_p20_h160_seed44_20260909`; checkpoint`checkpoints/epoch_3`, summary/predictions under`eval_validation/epoch_3`, expected diagnostics`cocoop_diagnostics.jsonl`, report`train_report.json`.

### 2026-09-10 - PathVQA QDPT causal Prompt placement controls seed44

- Exact experiments: `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_all_after_visual_seed44_20260910` and `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_reversed_sandwich_seed44_20260910`; implementation commit `74e87a9`. Both use PathVQA train19,654 and complete official Validation6,259 /832 image clusters, seed/data seed44/42,3 epochs and epoch3-only evaluation. Each retains the final D768 QDPT parameters, initialization, optimizer and exactly **7,805,184** trainable parameters; only the causal order of the static `P20`, visual tokens and dynamic `Z10` changes.
- All-after-visual order `[Visual; P20; Z10; Question]`: Overall **59.2587**, image-clustered95% CI **[57.8173,60.7160]**; Yes/No / Free-form **91.6160 /26.9943**. Per question type: how12.4031, other14.2857, what21.3501, when0.0000, where69.1932, why4.7619 and yes/no91.6160. Training runtime6,447.48s and train loss11.51436.
- Reversed-sandwich order `[Z10; Visual; P20; Question]`: Overall **58.1243**, image-clustered95% CI **[56.6195,59.4721]**; Yes/No / Free-form **91.4880 /24.8564**. Per question type: how9.3023, other14.2857, what19.7802, when0.0000, where63.5697, why4.7619 and yes/no91.4880. Training runtime6,463.69s and train loss11.58449.
- Reference sandwich `[P20; Visual; Z10; Question]` is **60.7765/92.7360/28.9100**. Therefore all-after loses **1.5178 Overall** and reversed sandwich loses **2.6522 Overall**. The result is not a generic benefit from placing Prompt tokens near the question: static `P20` works best before vision, where causal visual tokens can read the domain/task prior, while question-conditioned evidence `Z10` works best after vision and immediately before the question. Reversing these functional positions is the strongest failure and mainly damages Free-form and spatial `where` accuracy.
- Output paths: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_all_after_visual_seed44_20260910` and `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_reversed_sandwich_seed44_20260910`; checkpoints`checkpoints/epoch_3`, summaries/predictions under`eval_validation/epoch_3`, reports`train_report.json`.

### 2026-09-10 - PathVQA Static Visual Prompt Layer17 V20 seed44

- Exact experiment: `pathvqa_static_visual_prompt_l17_v20_seed44_20260910`; implementation commit `6bb24d8`. PathVQA train19,654 and complete official Validation6,259 /832 image clusters, seed/data seed44/42,3 epochs and epoch3-only evaluation.
- Controlled method: train only20 static visual Prompt tokens (`V20 x 1024`) at LR1e-4, inserted before frozen visual Block17 and stripped immediately after the single Block execution. There is no LLM-side Prompt, question-conditioned retrieval, dynamic Prompt or other trainable branch. Trainable parameters are exactly **20,480**.
- Validation Overall: **35.9482**, image-clustered95% CI **[34.7755,37.0823]**; Yes/No / Free-form **68.7040 /3.2865**. Per question type: how2.3256, other0.0000, what3.1005, when0.0000, where5.1345, why0.0000 and yes/no68.7040.
- Training runtime6,000.28s and train loss36.30333. The result is only about1.18 points above Frozen Base34.77 and open-answer performance is nearly absent. Static visual insertion alone cannot provide the answer-space/task adaptation required by generative PathVQA; this is direct evidence against treating visual-encoder Prompting as a substitute for the LLM interface.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/prompt_tuning/pathvqa_static_visual_prompt_l17_v20_seed44_20260910`; checkpoint`checkpoints/epoch_3`, summary/predictions under`eval_validation/epoch_3`, report`train_report.json`.

### 2026-09-10 - PathVQA Dual Static Prompt Layer17 V20 + LLM P20 seed44

- Exact experiment: `pathvqa_dual_static_prompt_l17_v20_p20_seed44_20260910`; implementation commit `6bb24d8`. PathVQA train19,654 and complete official Validation6,259 /832 image clusters, seed/data seed44/42,3 epochs and epoch3-only evaluation.
- Controlled method: combine the same Layer17 static visual `V20 x 1024` Prompt at LR1e-4 with an independent static LLM `P20 x 2560` Prompt at LR0.3. There is no question-conditioned retrieval or dynamic Prompt. Trainable parameters are exactly **71,680**.
- Validation Overall: **54.6253**, image-clustered95% CI **[53.1355,55.9817]**; Yes/No / Free-form **88.3840 /20.9636**. Per question type: how5.4264, other7.1429, what16.6405, when0.0000, where54.7677, why4.7619 and yes/no88.3840.
- Training runtime6,111.87s and train loss12.78837. Dual Static recovers the static LLM Prompt regime but is essentially tied with or slightly below the existing Static LLM Prompt Validation result around54.87. Consequently QDPT/Sandwich gains cannot be explained by merely adding trainable tokens to both modalities: the question-conditioned visual retrieval and correctly positioned dynamic evidence Prompt are the substantive additions.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/prompt_tuning/pathvqa_dual_static_prompt_l17_v20_p20_seed44_20260910`; checkpoint`checkpoints/epoch_3`, summary/predictions under`eval_validation/epoch_3`, report`train_report.json`.

### 2026-09-10 - PathVQA QDPT-Lite D768/R256 Sandwich seed44

- Exact experiment: `pathvqa_qdpt_lite_d768_r256_question_q10_l17_p20_s8_av10_sandwich_seed44_20260910`; implementation commit `021e2fc`. PathVQA train19,654 and complete official Validation6,259 /832 image clusters, seed/data seed44/42,3 epochs and epoch3-only evaluation.
- Controlled change: preserve the60.7765 Dense D768 Sandwich architecture exactly, including `[P20; Visual; Z10; Question]`, question-attention-pooled Q10, Layer17 full visual K/V, CA768/16, static `S8+A_v10`, all learning rates and zero-initialized final output. Only replace the dynamic text Prompt head `768->768->2560` with `768->256->2560`, decoupling evidence widthD768 from output-head rankR256.
- Trainable parameters: **6,100,736**, down1,704,448 / **21.84%** from Dense D768's7,805,184 and below Full-Attention LoRA-r8's7,077,888. Training completed1,845 steps in6,616.32s with train loss11.66845; the parameter reduction produces no meaningful runtime saving.
- Validation Overall: **57.4213**, standalone image-clustered95% CI **[55.9702,58.8170]**; Yes/No / Free-form **90.7840 /24.1544**. Per question type: how14.7287, other14.2857, what19.4270, when0.0000, where58.6797, why4.7619 and yes/no90.7840; Free-form question-type macro18.6472.
- Relative to the exact Dense Sandwich seed44 reference60.7765/92.7360/28.9100, R256 loses **3.3552 Overall**,1.9520 Yes/No and4.7556 Free-form, violating the predeclared maximum1-point loss by a wide margin. `where` falls from76.0391 to58.6797 (-17.3594), showing that the compressed output head particularly loses the ability to express spatially selected evidence. Paired significance has not been computed.
- Diagnostics rule out a detached branch or numerical collapse: Validation Cross-Attention delta/query is1.8186, text delta/anchor0.3954, text delta norm56.29, Workspace norm49.02 and visual-attention entropy0.5635. Slot pairwise cosine is0.8065, consistent with a more redundant evidence representation after output compression. The result is numerically almost identical to CoCoOp-style57.4053 despite retaining question-guided visual retrieval, indicating that an R256 output bottleneck prevents the retrieved evidence from being translated into sufficiently rich LLM-space Prompt tokens.
- Conclusion: reject R256 as both the main model and a claimed near-lossless Lite endpoint. Permanently retain the7.805M Dense D768 Sandwich as the final method; do not runR160, additional output ranks or R256 seeds. R256 may appear only as a negative output-rank capacity control demonstrating that retrieval width and LLM-space expression capacity cannot be compressed independently without substantial loss.
- Evaluation timing: TTFT mean0.053372s, weighted TPOT0.020022s/token,49.946 decode tokens/s and request mean0.107931s.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_lite_d768_r256_question_q10_l17_p20_s8_av10_sandwich_seed44_20260910`; checkpoint`checkpoints/epoch_3`, summary/predictions under`eval_validation/epoch_3`, diagnostics in the evaluation summary and report`train_report.json`.
### 2026-09-10 - PathVQA final Dense D768 Sandwich seed45 Validation

- Exact experiment: `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed45_20260910_1`; implementation commit `02d6fb8`. PathVQA official train19,654 and complete Validation6,259 /832 image clusters, model/data seeds45/42,3 epochs and epoch3-only full evaluation. Test is handled separately by the final suite.
- Configuration is identical to the frozen seed44 Dense D768 Sandwich reference: causal order `[P20; Visual; Z10; Question]`, question-attention-pooled Q10, Layer17 full visual K/V Cross-Attention at D768/16 heads, static visual `S8@3e-5+A_v10@1e-4`, P20 at LR0.3, no dynamic visual write and exactly **7,805,184** trainable parameters. The `_1` suffix only avoids the directory left by the previously interrupted launch; it is not a method variant.
- Validation Overall: **57.2935**, standalone image-clustered95% CI **[55.8265,58.6678]**. Yes/No / Free-form: **90.6880 /23.9949**. Per question type: how12.4031, other14.2857, what20.1727, when0.0000, where53.5452, why4.7619 and yes/no90.6880; Free-form question-type macro17.5281.
- Relative to the exact seed44 Sandwich result60.7765/92.7360/28.9087, seed45 changes Overall by **-3.4830**, Yes/No by-2.0480 and Free-form by-4.9138. The provisional two-seed Overall mean is59.0350; final mean and sample standard deviation must wait for seed46.
- Training completed1,845 steps in6,600.64s with train loss11.37548, versus11.20 for seed44. The divergence appears early rather than only at evaluation: at step200 the P20 norm is377.96 for seed45 versus442.93 for seed44, and at step600 it is486.29 versus637.45. Final P20 / text-anchor / Workspace norms are574.94/471.84/35.50 versus767.82/544.87/46.51 for seed44.
- Diagnostics rule out configuration drift, detachment and memory mismatch. Final visual-attention entropy is0.6467 versus0.5179 for seed44, indicating less concentrated visual selection; final sparse-visual gradient norm is0.01966 versus0.00467 and reaches0.04839 versus0.00537 at step1400, indicating sustained visual-side correction rather than a dead visual path. Question-query and visual-memory mismatch rates are zero, Cross-Attention remains active, and visual output/input ratios are nearly identical.
- Interpretation: the high-rate learned Prompt and Workspace entered a lower-scale, more diffuse-attention optimization trajectory from the first epoch and converged to a worse basin. This is evidence of initialization sensitivity, a known practical risk of soft-Prompt optimization, rather than evidence that the architecture or evaluator changed. The diagnostics identify the trajectory but do not establish a single causal tensor.
- Reporting rule: the paper must use the final three-seed mean +/- sample standard deviation as the primary performance claim. The60.7765 seed44 result may be shown as the best observed run only when explicitly labeled; it must not substitute for the multi-seed estimate. Discuss Prompt initialization sensitivity in Analysis/Limitations and retain the fixed model/data seed protocol for every baseline.
- Evaluation timing: TTFT mean0.053346s, weighted TPOT0.019514s/token,51.245 decode tokens/s and request mean0.106764s.
- Output path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed45_20260910_1`; checkpoint `checkpoints/epoch_3`, summary/predictions under `eval_validation/epoch_3`, report `train_report.json`.

### 2026-09-10 - Final Dense D768 Sandwich suite complete

- Frozen method for every run: `[P20; Visual; Z10; Question]`, question-attention-pooled Q10, Layer17 full visual K/V Cross-Attention at D768/16 heads, static visual `S8@3e-5+A_v10@1e-4`, P20 at LR0.3, no dynamic visual write,3 epochs, data seed42 and exactly **7,805,184** trainable parameters. Implementation/launcher commit is `02d6fb8`.
- PathVQA seed46 exact experiment `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed46_20260910`: complete Validation6,259 /832 clusters is **59.3865 Overall /91.5200 Yes-No /27.3452 Free-form**, clustered95% CI **[57.8619,60.7683]**. Per question type: how9.3023, other14.2857, what21.8995, when7.6923, where69.1932, why4.7619 and yes/no91.5200; Free-form macro21.1892. Training took6,657.57s with loss11.39112. Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed46_20260910`.
- PathVQA Validation three-seed final summary for seeds44/45/46: Overall **60.7765/57.2935/59.3865**, mean **59.1522 +/- 1.7528** sample standard deviation; Yes/No **92.7360/90.6880/91.5200**, mean **91.6480 +/- 1.0300**; Free-form **28.9087/23.9949/27.3452**, mean **26.7496 +/- 2.5107**. Seed46 lies between seed44 and45, confirming meaningful initialization sensitivity rather than a deterministic regression in seed45.
- PathVQA official Test was run exactly once on each frozen epoch3 checkpoint. Seed44: **60.4554/92.4152/28.4480**, clustered95% CI[59.1228,61.8567]. Seed45: **56.8983/90.1249/23.6223**, CI[55.5422,58.3625]. Seed46: **59.2945/91.4337/27.1075**, CI[58.0007,60.6613]. Every Test contains6,719 questions /858 image clusters.
- PathVQA Test three-seed summary: Overall mean **58.8827 +/- 1.8141** sample standard deviation; Yes/No mean **91.3246 +/- 1.1491**; Free-form mean **26.3926 +/- 2.4909**. Test preserves the same seed ordering as Validation, so there is no Test-driven model selection. Test output is under each experiment root at `eval_test/epoch_3/pathvqa_summary.json` and the adjacent predictions file.
- PathVQA Test per-type details: seed44 how12.9496, other50.0000, what22.9113, when0.0000, where69.3735, why4.5455; seed45 how9.3525, other50.0000, what19.8468, when0.0000, where52.6682, why0.0000; seed46 how10.7914, other27.7778, what21.2696, when0.0000, where70.9977, why4.5455. The unstable component is concentrated in Free-form and especially `where`, not only binary calibration.
- SLAKE seed44 exact experiment `slake_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260910`: official Test Overall **76.74**; CLOSED/OPEN **85.41/70.99**; KVQA/VQA **62.17/78.87**; EN/ZH **77.19/76.28**. Training took4,728.36s with loss3.50661. TTFT mean0.067697s, weighted TPOT0.020187s/token and49.537 decode tokens/s. Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260910`.
- SLAKE seed45 exact experiment `slake_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed45_20260910`: Test Overall **76.65**; CLOSED/OPEN **82.78/72.58**; KVQA/VQA **62.17/78.76**; EN/ZH **76.25/77.06**. Training took4,748.09s with loss3.45488. TTFT mean0.067541s, weighted TPOT0.019939s/token and50.152 decode tokens/s. Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed45_20260910`.
- SLAKE seed46 exact experiment `slake_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed46_20260910`: Test Overall **77.46**; CLOSED/OPEN **83.85/73.21**; KVQA/VQA **62.17/79.69**; EN/ZH **79.08/75.80**. Training took4,743.44s with loss3.60071. TTFT mean0.068100s, weighted TPOT0.020146s/token and49.637 decode tokens/s. Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed46_20260910`.
- SLAKE three-seed final summary: Overall **76.74/76.65/77.46**, mean **76.95 +/- 0.44** sample standard deviation; CLOSED **84.01 +/- 1.32**; OPEN **72.26 +/- 1.14**; KVQA **62.17 +/- 0.00**; VQA **79.11 +/- 0.51**; EN **77.51 +/- 1.44**; ZH **76.38 +/- 0.64**. Mean training runtime is4,739.96s, TTFT0.067779s and weighted TPOT0.020091s/token.
- Relative to the prior non-Sandwich QDPT-D768 SLAKE results77.65/76.70/76.74 (mean77.03), the final causal placement changes the mean by only **-0.08** and slightly reduces standard deviation0.54->0.44. The large PathVQA Sandwich gain therefore does not transfer as an accuracy gain to SLAKE, although the final architecture remains stable there.
- Relative to SLAKE Full-Attention LoRA-r8 mean81.82, final Sandwich QDPT remains **-4.87 Overall** while using7.805M versus7.078M trainable parameters. Preserve the established cross-dataset conclusion: PathVQA is near parity on mean Validation, while SLAKE favors LoRA decisively; do not claim universal superiority.
- Suite status: all eight execution units completed successfully: PathVQA seed45/46 train+Validation, all three PathVQA Tests, and SLAKE seed44/45/46 train+official Test. No further final-method training is scheduled.
### 2026-09-11 - Electrical final Dense D768 Sandwich seed47 interrupted run

- Exact experiment: `electrical_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed47_20260911`; implementation commit `1672f24`. Intended protocol is the private electrical dataset, model/data seeds47/42, final `[P20; Visual; Z10; Question]` Dense D768 Sandwich,3 epochs and one private fixed-holdout evaluation.
- Status: **incomplete, no score**. Training reached only step121/1875 (about6.5% of the planned optimizer steps, epoch approximately0.19) after about13.7 minutes. No `train_report.json`, epoch3 checkpoint, `electrical_summary.json` or `selected_result.tsv` exists.
- The captured `train.log` contains finite losses and diagnostics through step120, with loss15.05 at epoch0.16. All main branches are active and finite: shared Workspace gradient0.9993, sparse-visual gradient0.00513, text delta/anchor0.2157, visual-attention entropy0.9814, query and visual-memory mismatch rates0, and static visual write enabled. There is no Python traceback, CUDA exception or model assertion in the captured tail.
- Interpretation: the process ended externally at step121 rather than completing or producing an evaluable checkpoint. Current evidence cannot distinguish SSH/session termination, container lifecycle interruption or a system-level kill; it does not support an architecture or dataset-performance conclusion. Rerun the same target from the beginning when the server is stable, without changing the dataset or hyperparameters.
- Partial output path: `/root/autodl-tmp/Qwen3-VL-modify-test/electrical/outputs/qdpt/electrical_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed47_20260911`; partial `train.log` only.
### 2026-09-11 - PathVQA Static Prompt P20 seed stability study

- Exact completed experiments: existing `pathvqa_prompt_tuning_len20_seed44_20260827` plus new `pathvqa_prompt_tuning_len20_seed45_20260911` and `pathvqa_prompt_tuning_len20_seed46_20260911`; seed45/46 launcher commit `33c0e0d`. All use PathVQA train19,654 and complete official epoch3 Validation6,259, model seeds44/45/46, fixed data seed42,3 epochs, P20 at LR0.3, frozen Qwen3-VL and **51,200** trainable Prompt parameters. No new Test or SLAKE evaluation was run.
- Validation seeds44/45/46: Overall **54.8650/55.0567/55.3124**; Yes/No **89.5680/88.5120/88.7680**; Free-form **20.2616/21.6975/21.9528**.
- Three-seed summary: Overall **55.0780 +/- 0.2244** sample standard deviation with range0.4474; Yes/No **88.9493 +/- 0.5509** with range1.0560; Free-form **21.3040 +/- 0.9117** with range1.6912.
- Training seeds44/45/46: runtime5,969.59/5,931.72/5,899.17s and train loss12.05921/12.29612/12.06836. Mean runtime is5,933.49s. The supplied reports use an older schema with `total_trainable_parameters=null`; the exact51,200 count is independently determined by P20x2,560 and the existing parameter audit, not inferred as zero.
- Pre-registered continuation rule was Static Prompt Overall sample std>=0.5 or range>=1.0. Neither criterion is met (0.2244 and0.4474), so **do not run CoCoOp-style seed45/46** for the proposed universal Prompt-instability claim.
- Interpretation: Static Prompt has somewhat larger component variation than LoRA-r8, especially Free-form, but its aggregate Overall is stable because Yes/No and Free-form shifts partially offset. The current evidence therefore rejects the strong statement that Prompt methods are generally highly seed-sensitive. Final QDPT Sandwich's1.7528 Overall standard deviation is method-specific evidence associated with its higher-capacity conditional retrieval/Prompt pathway, not a demonstrated universal property of Prompt tuning.
- Paper use: a stability table may still report LoRA-r8, Static Prompt and QDPT side by side, but the defensible conclusion is a stability cost of the current conditional QDPT design. Do not frame Static Prompt as unstable or use its subgroup variance to conceal stable Overall performance.
- Output roots: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/prompt_tuning/pathvqa_prompt_tuning_len20_seed44_20260827`, `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/prompt_tuning/pathvqa_prompt_tuning_len20_seed45_20260911` and `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/prompt_tuning/pathvqa_prompt_tuning_len20_seed46_20260911`; checkpoints and summaries under `checkpoints/epoch_3` and `eval_validation/epoch_3`.
### 2026-09-11 - PathVQA CoCoOp-style conditional Prompt seed stability study

- Exact completed experiments: existing `pathvqa_cocoop_style_p20_h160_seed44_20260909` plus new `pathvqa_cocoop_style_p20_h160_seed45_20260911` and `pathvqa_cocoop_style_p20_h160_seed46_20260911`; seed45/46 launcher commit `5415543`. All use PathVQA train19,654 and complete official epoch3 Validation6,259, model seeds44/45/46, fixed data seed42,3 epochs, frozen Qwen3-VL, image-conditioned CoCoOp-style P20/H160, Prompt/Meta-Net LR0.3/3e-4 and exactly **873,120** trainable parameters. No Test or SLAKE evaluation was run.
- Validation seeds44/45/46: Overall **57.4053/56.3988/55.1366**; Yes/No **89.8560/89.5040/88.7040**; Free-form **25.0479/23.3886/21.6656**.
- Three-seed summary: Overall **56.3136 +/- 1.1367** sample standard deviation with range2.2687; Yes/No **89.3547 +/- 0.5903** with range1.1520; Free-form **23.3674 +/- 1.6913** with range3.3823. The best-seed Overall exceeds the mean by1.0917 points.
- Training seeds44/45/46: runtime5,962.33/6,043.88/6,017.43s and train loss11.74265/11.76250/12.05253. Mean runtime is6,007.88s. Seed46's lower Validation accompanies a higher final aggregate train loss, while seed44 and45 have nearly identical train loss despite a1.0065-point score gap; train loss alone is therefore not a reliable seed-selection signal.
- Stability comparison on the same PathVQA Validation protocol: LoRA-r8 Overall std0.0666, Static Prompt P20 std0.2244, CoCoOp-style std1.1367 and final QDPT Sandwich std1.7528. Corresponding known ranges are approximately0.14,0.4474,2.2687 and3.4830. This ordered pattern supports increasing initialization sensitivity from weight-space adaptation/static Prompting to image-conditioned and question-guided dynamic Prompting in the evaluated setup.
- Interpretation boundary: the evidence supports **conditional dynamic Prompt methods can be substantially more seed-sensitive under this frozen Qwen3-VL protocol**. It does not prove that every dynamic Prompt method, dataset or backbone is unstable, nor that parameter count alone causes variance. CoCoOp-style and QDPT differ in conditioning source and architecture, and only three seeds are available.
- Paper use: include a dedicated seed-stability table with per-seed Overall, mean +/- sample std, range and best-minus-mean for LoRA-r8, Static Prompt P20, CoCoOp-style and QDPT Sandwich. Present stability as an explicit cost of sample-conditioned Prompt adaptation and recommend multi-seed reporting; do not use the finding to justify reporting only QDPT seed44.
- Output roots: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/cocoop/pathvqa_cocoop_style_p20_h160_seed44_20260909`, `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/cocoop/pathvqa_cocoop_style_p20_h160_seed45_20260911` and `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/cocoop/pathvqa_cocoop_style_p20_h160_seed46_20260911`; checkpoints and summaries under `checkpoints/epoch_3` and `eval_validation/epoch_3`.

### 2026-09-12 - Private electrical final Prompt comparison seed47

- Protocol: all three completed runs use the unchanged private electrical training data and fixed holdout, model seed47, data seed42,3 epochs, frozen Qwen3-VL backbone and one epoch3 evaluation. Every evaluation reports **954 evaluated /18 skipped** samples. Only aggregate accuracy was supplied; no answer-type breakdown, paired predictions or confidence interval is currently available.
- Static Prompt exact experiment `electrical_static_prompt_p20_seed47_20260911`: P20 at LR0.3, **51,200** trainable parameters, private accuracy **70.06**. Output: `/root/autodl-tmp/Qwen3-VL-modify-test/electrical/outputs/prompt_tuning/electrical_static_prompt_p20_seed47_20260911`; summary under `eval_private/epoch_3/electrical_summary.json`.
- CoCoOp-style exact experiment `electrical_cocoop_style_p20_h160_seed47_20260911`: image-conditioned P20/H160 with Prompt/Meta-Net LR0.3/3e-4, **873,120** trainable parameters, private accuracy **70.88**. Output: `/root/autodl-tmp/Qwen3-VL-modify-test/electrical/outputs/cocoop/electrical_cocoop_style_p20_h160_seed47_20260911`; summary under `eval_private/epoch_3/electrical_summary.json`.
- QDPT exact experiment `electrical_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed47_20260911_1`: final `[P20; Visual; Z10; Question]` Dense D768 Sandwich, **7,805,184** trainable parameters, private accuracy **71.91**. This is a fresh completed run and does not replace the separately recorded interrupted directory without suffix. Output: `/root/autodl-tmp/Qwen3-VL-modify-test/electrical/outputs/qdpt/electrical_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed47_20260911_1`; summary under `eval_private/epoch_3/electrical_summary.json`.
- Descriptive comparison: `Static70.06 < CoCoOp70.88 < QDPT71.91`. QDPT improves by **+1.85** over Static Prompt and **+1.03** over CoCoOp-style. Relative to the previously supplied electrical Full-Attention LoRA-r8 score **70.69**, QDPT is **+1.22**, CoCoOp-style is +0.19 and Static Prompt is -0.63. These are point-estimate differences only and must not be called statistically significant without paired predictions.
- Interpretation: the private electrical task is dominated by visual discrimination with a closed multiple-choice answer space. This matches QDPT's inductive bias: the question retrieves relevant evidence from frozen visual K/V and converts it into an LLM-side dynamic Prompt. The result supports task affinity when the backbone already contains the required visual distinctions and the main bottleneck is question-conditioned evidence selection.
- Claim boundary: do **not** present this result as universal superiority over LoRA or as proof that QDPT adds domain knowledge or stronger multi-step reasoning. Closed answer choices, answer priors and reduced language-generation difficulty may all favor Prompt-space adaptation. Together with near parity on PathVQA and the clear LoRA advantage on SLAKE, the defensible conclusion is that Prompt-space and weight-space adaptation have different task preferences. The proposed mechanism remains an evidence-supported explanation, not an isolated causal finding, until question type, visual difficulty and reasoning depth are evaluated separately.

### 2026-09-13 - PathVQA final-method inference timing audit

- Protocol: arithmetic mean over the three completed seed44/45/46 epoch3 full-Validation summaries, matching the averaging convention used in the paper's SLAKE efficiency table. These are model-generated timing measurements after the evaluator's warmup, not end-to-end service latency.
- Final QDPT Dense D768 Sandwich TTFT is **0.050772/0.053346/0.053235s**, mean **0.052451s**. Weighted TPOT is **0.015883/0.019514/0.020065s/token**, mean **0.018487s/token**.
- Full-Attention LoRA-r8 TTFT is **0.057489/0.057278/0.057887s**, mean **0.057551s**. Weighted TPOT is **0.030869/0.029367/0.030856s/token**, mean **0.030364s/token**.
- Descriptively, QDPT has an8.86% lower mean TTFT and39.12% lower mean TPOT than LoRA-r8 under these recorded PathVQA runs. Hardware/software conditions must be reported as matched before treating this as a controlled speed claim; otherwise present the values as observed efficiency measurements.
- Source summaries: QDPT `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260909`, `...seed45_20260910_1`, `...seed46_20260910`; LoRA `pathvqa_lora_full_model_attention_r8_seed44_20260830`, `...seed45_20260904`, `...seed46_20260904`, each under `eval_validation/epoch_3/pathvqa_summary.json`.

### 2026-09-13 - PathVQA unified V20 Sandwich RNG-controlled negative result

- Exact logical experiment: `pathvqa_qdpt_d768_question_q10_l17_p20_unified_v20_sandwich_rng_control_seed44`; launcher commit `cb08fc8`. The controlled change retains the final Dense D768 Sandwich order `[P20; Visual; Z10; Question]`, seed/data seed44/42,3 epochs, question-guided Layer17 visual K/V and downstream RNG initialization, but replaces `S8@3e-5+A_v10@1e-4` with one `V20@1e-4` table. Trainable parameters are **7,807,232**.
- User-reported displayed Validation Overall is approximately **55.70**. The exact unrounded score, Yes/No, Free-form, diagnostics and output path have not yet been supplied and remain pending extraction from the completed summary; do not invent or use unavailable breakdowns.
- Relative point estimates: approximately **-5.08** versus the retained `S8+A_v10` Sandwich seed44 score60.7765, and approximately **-2.82** versus the RNG-controlled unified V20 non-Sandwich score58.5237. By contrast, Sandwich improved the retained dual-rate parameterization from59.5622 to60.7765 (+1.2143), so the position change has opposite effects under the two visual-Prompt parameterizations.
- Interpretation: this result rejects unified V20 as a cosmetic replacement in the final Sandwich architecture. Because downstream initialization is controlled and the text placement flag adds no parameters, the large loss is evidence of a real optimization interaction between visual-Prompt parameterization and causal LLM Prompt placement, not the earlier accidental global-RNG drift. A plausible mechanism is that Sandwich strengthens joint gradients through the visual-token region: the dual-rate form preserves an8-token slow visual basis while allowing the10-token anchor to adapt faster, whereas a single all-fast V20 table has no stable subspace and may co-adapt destructively with P20/Z10. This mechanism is provisional until norm, gradient and attention diagnostics are compared.
- Reporting boundary: the experiment still changes table structure, token count18->20 and learning-rate assignment together, so it cannot isolate which factor causes the interaction. It is sufficient to justify retaining the empirical `S8@3e-5+A_v10@1e-4` design, but not sufficient to claim that unified visual Prompts or20 visual tokens are generally harmful.

### 2026-09-13 - PathVQA GRASP-Qwen embedding-init and dual-LR rescue seed44

- Exact logical experiment: `pathvqa_grasp_qwen_adapted_embedding_init_dual_lr_n4_h512_seed44`; commit `bae0965`. Dataset PathVQA, model/data seeds44/42, three epochs and complete official Validation evaluation.
- Controlled change relative to the formally usable GRASP reimplementation39.7508: retain N=4, h=512, Entmax-1.5, one generated global Prompt, raw-question frozen-LLM encoding, post-merger spatial blocks and paper-order `[Prompt, Visual, Question]`; initialize the four Prompt prototypes from frozen Qwen token-embedding rows instead of Gaussian std0.02, optimize prototypes at LR0.3 with no weight decay, and retain routing projections at LR1e-4 with weight decay0.01.
- User-reported displayed Validation Overall: **42.40**, approximately **+2.65** over the Gaussian/shared-LR paper-order reproduction39.7508. Exact unrounded score, Yes/No, Free-form, diagnostics and output path have not yet been supplied; do not invent unavailable values.
- Correction from the subsequently supplied complete summary and diagnostics: exact Validation is **42.4029 Overall /76.2240 Yes-No /8.6790 Free-form**. Training runtime is6,626.07s, aggregate train loss13.88198, and epoch-window loss means are16.4003/13.0640/12.2497. Prompt norm grows from a first-window mean33.86 to80.40 in the last window and87.95 in the final record; prototype norm grows72.16->180.86 and ends183.07. Late routing is active and sparse: entropy0.5082, max weight0.6921 and zero fraction0.2708; projection gradient norm remains about0.9997 while Prompt-prototype gradient norm is0.0220. Output path: `pathvqa/outputs/grasp/pathvqa_grasp_qwen_adapted_embedding_init_dual_lr_n4_h512_seed44_20260913`.
- Interpretation: Qwen-native Prompt initialization and Prompt-specific optimization recover a measurable part of the loss, confirming that direct transfer of the original Prompt scale was mismatched. The result remains approximately12.47 points below Static Prompt54.8650, so scale mismatch is not the dominant explanation for GRASP's failure. Do not treat longer training or a still larger Prompt LR as the next isolated test.
- Next controlled hypothesis: quantify and then remove or gate the fixed 2D sinusoidal position encoding. Its theoretical norm at hidden size2560 is about35.8, while the prior run logged position-added visual-block norms around40-44, so routing K may be dominated by coarse quadrant identity rather than image semantics. Keep the successful embedding initialization and dual-LR settings fixed while changing only positional encoding.

### 2026-09-13 - PathVQA GRASP-Qwen no-position rescue seed44

- Exact logical experiment: `pathvqa_grasp_qwen_adapted_no_position_n4_h512_seed44`; commit `9994463`. Dataset PathVQA, model/data seeds44/42, three epochs and complete official Validation evaluation.
- Controlled change relative to the42.40 embedding-init/dual-LR rescue: remove the fixed 2D sinusoidal position encoding before the visual key projection. N=4, h=512, Entmax-1.5, embedding-row Prompt initialization, Prompt/projection LR0.3/1e-4, one global Prompt, raw-question frozen-LLM encoding and `[Prompt, Visual, Question]` remain unchanged.
- User-reported displayed Validation Overall: **43.92**, approximately **+1.52** over42.40 and **+4.17** over the formally usable39.7508 reproduction. Exact unrounded score, answer-type breakdowns, diagnostics and output path remain pending extraction.
- Correction from the subsequently supplied complete summary and diagnostics: exact Validation is **43.9208 Overall /79.0400 Yes-No /8.9024 Free-form**. Relative to42.4029, the exact changes are **+1.5179 Overall /+2.8160 Yes-No /+0.2234 Free-form**, so almost all benefit is confined to closed questions. Training runtime is6,701.01s, aggregate train loss14.05970, and epoch-window loss means are16.5403/13.2487/12.4597. Prompt norm grows36.98->79.03 and ends91.17; prototype norm grows77.92->172.46 and ends174.45, ruling out continued Prompt under-scaling. Raw visual-block norm is21.98 early,22.54 late and23.05 final. Late routing becomes less sparse than the position-encoded run (entropy0.6294 vs0.5082, max weight0.5959 vs0.6921, zero fraction0.1458 vs0.2708), although both final records select one dominant region at about0.81. Projection gradients remain near the clipping boundary at0.9995 while Prompt gradients remain nonzero0.0287. Output path: `pathvqa/outputs/grasp/pathvqa_grasp_qwen_adapted_no_position_n4_h512_seed44_20260913`.
- Refined interpretation: removing position encoding improves Validation despite slightly worse training loss throughout, suggesting reduced positional overfitting rather than easier optimization. It does not repair open-form generation. The decreasing epoch-window loss and continued Prompt-norm movement mean convergence is not established, but epoch3-only evaluation cannot show whether Validation is still rising. Evaluate the already-saved epoch1/2 checkpoints before committing to a new10/20-epoch run.
- Subsequent checkpoint evaluation correction: the existing no-position epoch1/2/3 checkpoints score **42.1793/43.4414/43.9208 Overall**, **77.6320/78.5280/79.0400 Yes-No**, and **6.8283/8.4556/8.9024 Free-form**. Epoch2 per-type scores are how5.4264, other0, what5.7300, when0, where27.3839, why0 and yes/no78.5280; image-clustered95% CI[42.06,44.77]. Epoch2 timing is TTFT0.068446s and TPOT0.020189s/token. Output summaries are under the same run's `eval_validation/epoch_1`, `epoch_2`, and `epoch_3` directories.
- Convergence conclusion: Validation improves monotonically through epoch3, directly confirming incomplete three-epoch convergence for this adapted GRASP configuration. Gains diminish from **+1.2621** Overall between epochs1-2 to **+0.4794** between epochs2-3; Free-form gains similarly diminish from+1.6273 to+0.4468. Thus additional optimization is justified before structural modification, but the observed trajectory alone does not support expecting the approximately10.95-point gap to Static Prompt to disappear. A fresh longer run is required because the existing three-epoch linear scheduler has already decayed its LR to zero; continuing epoch3 in place is not a valid long-horizon test.

### 2026-09-13 - Failed PathVQA GRASP-Qwen no-position 10-epoch marathon seed44

- Exact logical experiment: `pathvqa_grasp_qwen_adapted_no_position_marathon10_n4_h512_seed44`; commit `f5cbaae`. Controlled change from the43.9208 three-epoch run was only the training/scheduler horizon from3 to10 epochs; architecture, initialization, Prompt/projection LR0.3/1e-4, seed and data protocol were retained.
- Status: **failed due to numerical divergence and manually stopped at approximately epoch5.82**. No valid final Validation score exists. Checkpoints epoch1-5 were written, but epoch2 onward must be treated as suspect or invalid until finite-state inspection; epoch3-5 were created after diagnostics had already become non-finite.
- Evidence: epoch1 logged loss mean17.4057, first66.75, last14.89 and minimum13.59. During epoch2 the logged mean becomes122,678.63, followed by reported zeros; epochs3-6 show only0.0. The zeros are not convergence: latest diagnostics at epochs5.692-5.822 report NaN global Prompt norm, prototype norm, routing entropy, maximum weight and both Prompt/projection gradient norms.
- Interpretation: extending the linear scheduler horizon while retaining Prompt LR0.3 materially changes the optimization exposure. In the successful3-epoch run, the high Prompt LR decays to zero quickly; in the10-epoch schedule it remains large for much longer and causes runaway non-finite Prompt/projection state during epoch2. This failed run does not test whether longer GRASP training improves generalization. Any replacement must reduce or cap cumulative Prompt optimization, add explicit finite guards, and preserve the failed run as negative evidence rather than evaluating corrupted checkpoints.
- Interpretation: the large fixed positional signal was mildly harmful, supporting the hypothesis that it competed with real visual semantics, but it was not the dominant failure. Even after correcting initialization/optimization scale and removing positional encoding, GRASP remains approximately10.95 points below Static Prompt54.8650. The next high-value axis is output capacity: test whether collapsing all routed regions into one static-prototype Prompt is the principal bottleneck before spending on a long schedule.

### 2026-09-14 - PathVQA Sandwich Learned Query capacity-matched control seed44

- Exact logical experiment: `pathvqa_qdpt_d768_learned_q10_l17_p20_s8_av10_sandwich_seed44`; implementation commit `a4bf3ba`. Dataset PathVQA, model/data seeds44/42, three epochs and complete official Validation evaluation.
- Controlled change: retain the final D768 Sandwich architecture, Layer17 full visual K/V, Cross-Attention, `P20`, `Z10`, `S8+A_v10`, `[P20; Visual; Z10; Question]`, optimizer and all learning rates. Replace only the question-attention-pooling score projection `workspace_text_score_projection[10,2560]` with an equal-size learned static query table `learned_static_query[10,2560]`. Both methods contain exactly **7,805,184** trainable parameters.
- User-reported displayed Validation result: **59.05 Overall /90.88 Yes-No /27.31 Free-form**. The exact unrounded summary values, diagnostics and unique output directory have not yet been supplied; record these displayed values as rounded and do not invent unavailable precision or significance statistics.
- Same-seed comparison with the final question-guided Sandwich result60.7765/92.7360/28.9087 gives approximate changes of **-1.73 Overall /-1.86 Yes-No /-1.60 Free-form**. Because parameter count, visual evidence path, Prompt placement and training protocol are matched, the loss cannot be attributed to lower adapter capacity or removal of visual K/V access.
- Conclusion: current-question conditioning contributes beyond a generic learned-query visual aggregator in the final architecture. The control strengthens the mechanism claim that QDPT performs question-directed retrieval rather than merely adding learned visual queries. This is a single-seed controlled effect, so report it as a seed44 capacity-matched ablation and do not claim cross-seed stability or statistical significance until paired predictions are available.

### 2026-09-14 - PathVQA Prompt-placement controls seeds45/46

- Exact experiments: `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_all_after_visual_seed45_20260914`, `...all_after_visual_seed46_20260914`, `...reversed_sandwich_seed45_20260914`, and `...reversed_sandwich_seed46_20260914`; launcher commit `015987f`. All use PathVQA train19,654 and complete official Validation6,259, model seeds45/46, data seed42,3 epochs and epoch3-only evaluation. Each retains the final D768 modules, initialization rule, optimizer and exactly **7,805,184** trainable parameters; only the causal order of `P20`, visual tokens and `Z10` differs.
- All-after-visual `[Visual; P20; Z10; Question]`: seed45 **58.4119 Overall /90.6880 Yes-No /26.2285 Free-form**, runtime6,585.32s, train loss12.42346; seed46 **59.2267/91.7440/26.8028**, runtime6,584.64s, train loss11.57391. Output roots: `pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_all_after_visual_seed45_20260914` and `...seed46_20260914`.
- Reversed Sandwich `[Z10; Visual; P20; Question]`: seed45 **58.4119 Overall /90.5920 Yes-No /26.3242 Free-form**, runtime6,576.38s, train loss11.37710; seed46 **59.0989/91.2320/27.0581**, runtime6,586.97s, train loss11.35209. Output roots: `pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_reversed_sandwich_seed45_20260914` and `...seed46_20260914`.
- Final diagnostics for all-after seed45/46 respectively: P20 norm614.59/649.13, text-anchor norm561.47/558.11, Z10 norm48.45/52.62, visual-attention entropy0.6850/0.5916, Cross-Attention delta/query1.0492/0.9807, text delta/anchor0.5494/0.5081, slot cosine0.6055/0.8203 and sparse-visual gradient0.00504/0.00568. Reversed seed45/46: P20 norm658.98/697.10, text-anchor norm613.80/567.66, Z10 norm46.70/44.00, entropy0.5528/0.5784, Cross-Attention delta/query1.1876/1.9119, text delta/anchor0.4322/0.7244, slot cosine0.7773/0.9063 and sparse-visual gradient0.02334/0.01175. All branches remain finite and active; these endpoints do not support a single monotonic norm-based explanation of accuracy.
- Three-seed aggregates using the previously completed seed44 controls: final Sandwich `[P20; Visual; Z10; Question]` is **59.1522 +/- 1.7528 Overall**, **91.6480 +/- 1.0300 Yes-No**, and **26.7496 +/- 2.5107 Free-form**; all-after is **58.9658 +/- 0.4799**, **91.3493 +/- 0.5763**, and **26.6752 +/- 0.3985**; reversed is **58.5450 +/- 0.5008**, **91.1040 +/- 0.4615**, and **26.0796 +/- 1.1211**. Values are mean +/- sample standard deviation; the final Sandwich statistics retain the existing calculation from the unrounded source summaries, whereas the newly supplied placement values are available only at four-decimal display precision.
- Same-seed Overall differences for all-after minus final Sandwich are **-1.5178/+1.1184/-0.1598** at seeds44/45/46; reversed minus final Sandwich gives **-2.6522/+1.1184/-0.2876**. The ranking therefore changes with initialization. The mean Sandwich advantages are only0.1864 over all-after and0.6071 over reversed, both much smaller than the Sandwich standard deviation.
- Correction to the seed44-only interpretation recorded on2026-09-10: Prompt placement clearly changes optimization and can have a large effect for one initialization, but the claim that the Sandwich ordering is stably superior is not supported. Retain Sandwich as the configuration selected on seed44 Validation before these replications and already frozen for final Test/SLAKE evaluation; switching now would be post-hoc selection. In the paper, describe placement as an initialization-sensitive design choice, report all three seeds, and do not claim a universal causal-order mechanism without paired multi-seed uncertainty analysis. Per-question-type and clustered-CI fields were not present in the supplied extraction.

### 2026-09-14 - PathVQA baseline official Test evaluation

- Protocol: inference-only evaluation of the frozen epoch3 checkpoints on the official PathVQA Test split. No model was retrained and no Test result was used for architecture or checkpoint selection. Each summary is stored under its experiment root at `eval_test/epoch_3/pathvqa_summary.json`.
- Static Prompt P20 exact experiments `pathvqa_prompt_tuning_len20_seed44_20260827`, `pathvqa_prompt_tuning_len20_seed45_20260911`, and `pathvqa_prompt_tuning_len20_seed46_20260911` obtain Test Overall **55.8268 / 55.9161 / 56.4965**, mean **56.0798 +/- 0.3636** sample standard deviation. Output roots are under `pathvqa/outputs/prompt_tuning/`. Only Overall and summary paths were supplied in this extraction; Yes/No, Free-form, per-type and clustered-CI values remain to be read from the saved summaries.
- CoCoOp-style P20/H160 exact experiments `pathvqa_cocoop_style_p20_h160_seed44_20260909`, `pathvqa_cocoop_style_p20_h160_seed45_20260911`, and `pathvqa_cocoop_style_p20_h160_seed46_20260911` obtain Test Overall **57.7467 / 57.0323 / 56.0798**, mean **56.9529 +/- 0.8363** sample standard deviation. Output roots are under `pathvqa/outputs/cocoop/`. Only Overall and summary paths were supplied in this extraction; other saved breakdowns remain pending extraction.
- Full-Attention LoRA-r8 exact experiments `pathvqa_lora_full_model_attention_r8_seed44_20260830` and `pathvqa_lora_full_model_attention_r8_seed45_20260904` obtain Test Overall **59.6815 / 59.6369**, provisional two-seed mean **59.6592 +/- 0.0315**. Their output roots are under `pathvqa/outputs/lora/`. Seed46 has an `eval_test_epoch_3.log` but no `eval_test/epoch_3/pathvqa_summary.json`, so it is **incomplete/failed and must not enter the three-seed aggregate** until separately diagnosed or rerun.
- For context, the already frozen QDPT Sandwich Test mean is **58.8827 +/- 1.8141**. The current complete three-seed Test means put Static Prompt at56.0798 and CoCoOp-style at56.9529; the provisional two-seed LoRA mean is59.6592. Final QDPT-versus-LoRA Test comparison remains pending LoRA seed46.
- Completion correction: `pathvqa_lora_full_model_attention_r8_seed46_20260904` has now completed inference-only evaluation on all **6,719 Test questions /858 image clusters** from its frozen epoch3 checkpoint. Displayed Test result is **59.67 Overall /91.6716 Yes-No /27.61 Free-form**. Per-type scores are how12.2302, other50.0000, what21.4520, when0.0000, where72.3898, why4.5455 and yes/no91.6716. Standalone image-clustered95% CI is **[58.29,61.09]**. Timing is TTFT mean0.055708s, p500.054451s and p950.056733s; TPOT0.030822s/token or32.444 tokens/s. Summary path: `pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r8_seed46_20260904/eval_test/epoch_3/pathvqa_summary.json`.
- The preceding provisional status is superseded. Using the displayed seed46 Overall precision, LoRA-r8 Test seeds44/45/46 are **59.6815 /59.6369 /59.67**, with approximate mean **59.6628 +/- 0.0232** sample standard deviation. Relative to QDPT Sandwich Test mean58.8827, the descriptive LoRA mean advantage is approximately **0.78 points**; exact final aggregation should use the unrounded seed46 value from the saved summary.

### 2026-09-14 - PathVQA Learned Query capacity-matched multi-seed completion

- Exact experiments: `pathvqa_qdpt_d768_learned_q10_l17_p20_s8_av10_sandwich_seed44_20260913`, `...seed45_20260914`, and `...seed46_20260914`; implementation/launcher commit `b9eb353`. All use PathVQA train19,654 and complete Validation6,259 /832 image clusters, model seeds44/45/46, data seed42,3 epochs and epoch3-only evaluation. Each has exactly **7,805,184** trainable parameters and differs from final QDPT only by replacing question-attention-pooled Q10 with a capacity-matched learned static `10x2560` query table while retaining the common `2560->768` projection, full Layer17 visual K/V and `[P20; Visual; Z10; Question]` order.
- Seed44: **59.0510 Overall /90.8800 Yes-No /27.3133 Free-form**, clustered95% CI[57.4992,60.5101]. Per type: how10.0775, other14.2857, what21.9388, when0, where68.7042, why4.7619 and yes/no90.8800. Runtime6,534.89s and train loss11.34226.
- Seed45: **57.7249 /90.1120 /25.4308**, CI[56.2102,59.1680]. Per type: how6.2016, other7.1429, what20.9184, when0, where62.3472, why0 and yes/no90.1120. Runtime6,328.49s and train loss11.50538.
- Seed46: **58.1882 /89.4080 /27.0581**, CI[56.8115,59.5986]. Per type: how6.9767, other14.2857, what21.8603, when7.6923, where68.2152, why0 and yes/no89.4080. Runtime6,321.64s and train loss11.45018.
- Learned Query three-seed Overall is **58.3214 +/- 0.6730** sample standard deviation. Against question-guided QDPT's matched seeds60.7765/57.2935/59.3865, Learned Query minus QDPT is **-1.7255/+0.4314/-1.1983**, with mean difference **-0.8308**. Question guidance wins on the mean and two of three seeds, but seed45 reverses the ordering; this supports an average benefit under matched capacity, not a seed-invariant mechanism claim.
- Output roots are under `pathvqa/outputs/dynamic_prompt/` using the exact experiment names above; summaries are at `eval_validation/epoch_3/pathvqa_summary.json`. Per the frozen plan, no Learned Query SLAKE or Test run is scheduled.

### 2026-09-14 - Controlled QDPT versus LoRA-r8 training-throughput benchmark

- Exact logical target: `pathvqa_qdpt_lora_training_throughput_benchmark`; commit `b9eb353`; NVIDIA GeForce RTX5090. Both methods use the same PathVQA data order, model/data seed44/42, BF16, dynamic resolution, microbatch1, gradient accumulation32 and effective batch32. Each receives20 optimizer-step warmup followed by100 timed optimizer steps; evaluation, checkpoint saving and method-specific diagnostics are disabled. The timed windows process the same **3,200 samples and1,202,879 post-merger visual tokens**.
- QDPT exact run `pathvqa_qdpt_d768_sandwich_throughput_w20_t100_seed44_20260914`: elapsed **438.0684s**,4.3807s/optimizer-step, **7.3048 samples/s** and2,745.87 visual tokens/s. CUDA peak allocated/reserved memory is **15.805/22.572 GiB**. Extrapolated pure three-epoch training time is2.245h. Report: `pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_sandwich_throughput_w20_t100_seed44_20260914/throughput_report.json`.
- Full-Attention LoRA-r8 exact run `pathvqa_lora_full_attention_r8_throughput_w20_t100_seed44_20260914`: elapsed **824.0839s**,8.2408s/optimizer-step, **3.8831 samples/s** and1,459.66 visual tokens/s. CUDA peak allocated/reserved memory is **19.317/24.914 GiB**. Extrapolated pure three-epoch training time is4.223h. Report: `pathvqa/outputs/lora/pathvqa_lora_full_attention_r8_throughput_w20_t100_seed44_20260914/throughput_report.json`.
- Under this controlled short-run protocol, QDPT is **1.881x faster**, uses **3.512 GiB /18.18% less peak allocated memory**, and uses2.342 GiB /9.40% less peak reserved memory. Identical sample and visual-token counts rule out workload volume as the source of the difference.
- Claim boundary: this supports a controlled training-throughput and memory-efficiency claim, not an exact wall-clock claim for complete runs. The2.245h/4.223h values are linear extrapolations that omit evaluation, checkpoint I/O and startup; report measured100-step throughput as primary and label extrapolated totals explicitly.

### 2026-09-21 - PathVQA QDPT seed44/45 inference-only P20 and text-anchor swaps

- Exact diagnostic target: `pathvqa_qdpt_stage1_component_swaps`; implementation commits `a2a1734`, `94a0c99` and `1d7d82d`. Dataset is the complete PathVQA Validation split with **6,259 questions /832 image clusters**. No training or checkpoint modification was performed. Each variant loads one complete final Dense D768 Sandwich epoch3 checkpoint as the receiver and replaces exactly one named tensor from the other seed checkpoint after strict shape auditing.
- Receiver seed44 baseline is **60.7765 Overall /92.7360 Yes-No /28.9087 Free-form**. `P20_45 + rest_44` obtains **54.2898 /85.8240 /22.8462**, changes **-6.4867 /-6.9120 /-6.0625**, and has variant-only/baseline-only correct counts170/576, exact McNemar `p=2.04e-52`, clustered paired95% CI **[-7.3829,-5.5346]**. Per-type changes include what-4.7096 and where-16.1369.
- `A_t10_45 + rest_44` obtains **53.0276 Overall /87.6480 Yes-No /18.5067 Free-form**, changes **-7.7488 /-5.0880 /-10.4020**, and has variant-only/baseline-only counts162/647, exact McNemar `p=2.35e-69`, paired95% CI **[-8.6717,-6.8249]**. Per-type changes include what-7.3783 and where-32.2738.
- Receiver seed45 baseline is **57.2935 Overall /90.6880 Yes-No /23.9949 Free-form**. `P20_44 + rest_45` obtains **55.0727 /88.2560 /21.9847**, changes **-2.2208 /-2.4320 /-2.0102**, and has variant-only/baseline-only counts167/306, exact McNemar `p=1.65e-10`, paired95% CI **[-2.9753,-1.4722]**. Its where score changes **+3.1785** despite the significant aggregate loss.
- `A_t10_44 + rest_45` obtains **49.3849 Overall /82.6880 Yes-No /16.1774 Free-form**, changes **-7.9086 /-8.0000 /-7.8175**, and has variant-only/baseline-only counts268/763, exact McNemar `p=1.39e-55`, paired95% CI **[-8.9320,-6.8881]**. Per-type changes include how-9.3023, what-6.0440 and where-19.0709.
- Interpretation: neither the better seed44 P20 nor its text anchor transfers as an independently superior component. All four cross-seed replacements significantly harm their receiver, and both directions of the text-anchor swap collapse by approximately7.8 points. This is evidence of strong seed-specific co-adaptation between the static/text-anchor coordinate system and the remaining trained generator, not evidence that one isolated donor tensor is the primary cause of the original3.483-point seed gap. The asymmetric P20 results and seed44-P20 where gain indicate some transferable spatial bias, but it is outweighed by broken global compatibility.
- Scope boundary: these swaps are causal compatibility diagnostics, not fair deployable models and not candidates for the main performance table. The subsequently added `Visual18` bidirectional swap was not present in this completed run and remains pending. Timing is not interpreted because matched isolated-GPU conditions were not established for this diagnostic run.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_stage1_component_swaps_seed44_45_20260921`; detailed paired reports are `receiver_seed44/conditioning_mismatch_comparison.{json,tsv}` and `receiver_seed45/conditioning_mismatch_comparison.{json,tsv}`.

### 2026-09-22 - PathVQA QDPT seed44/45 inference-only Visual18 swaps

- This is an explicit continuation of the 2026-09-21 component-swap diagnostic above; it does not revise or replace the four earlier P20/text-anchor results. Exact target: `pathvqa_qdpt_stage1_component_swaps`; Visual18 implementation commit `ca611c4`, restart-safe orchestration commit `67b46ea`. Dataset is the complete PathVQA Validation split with **6,259 questions /832 image clusters**. No training or checkpoint modification was performed. `Visual18` atomically replaces the complete Layer17 static visual Prompt state `S8+A_v10` from the other seed while retaining every other receiver parameter.
- Receiver seed44 baseline is **60.7765 Overall /92.7360 Yes-No /28.9087 Free-form**. `Visual18_45 + rest_44` obtains **60.4729 /92.8000 /28.2387**, changes **-0.3036 /+0.0640 /-0.6701**, and has variant-only/baseline-only correct counts24/43, exact McNemar `p=0.02712`, clustered paired95% CI **[-0.5578,-0.0474]**. Per-question-type changes are how-0.7752, other0, what-0.5102, when0, where-1.7115, why0 and yes/no+0.0640.
- Receiver seed45 baseline is **57.2935 Overall /90.6880 Yes-No /23.9949 Free-form**. `Visual18_44 + rest_45` obtains **57.0858 /90.6560 /23.6120**, changes **-0.2077 /-0.0320 /-0.3829**, and has variant-only/baseline-only correct counts39/52, exact McNemar `p=0.20817`, clustered paired95% CI **[-0.5212,+0.0960]**. Per-question-type changes are how-1.5504, other0, what-0.3925, when0, where0, why0 and yes/no-0.0320.
- The mean absolute Overall loss of the two Visual18 swaps is only **0.2556**, versus **4.3537** for P20 and **7.8287** for the text anchor. The original seed44-seed45 baseline gap is3.4830 points; after opposite-direction Visual18 replacement it remains3.3871 points, approximately97% of the original gap. Thus the endpoint `S8+A_v10` visual Prompt is highly portable across these seeds and does not explain the observed multi-seed instability. The one direction with `p<0.05` has a small practical effect concentrated in Free-form and does not alter this conclusion.
- Interpretation boundary: this result excludes the saved Visual18 tensors as the dominant source of the seed gap; it does not prove that all visual adaptation or the frozen visual representation is irrelevant. The severe P20/anchor swap losses still establish text-side coordinate co-adaptation, but swaps alone cannot identify P20, anchor or the remaining dynamic generator as the unique causal source. Continue diagnosis on text-side initialization and joint optimization trajectory rather than redesigning the lightweight visual Prompt first.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_stage1_component_swaps_seed44_45_20260921_1`; detailed paired reports are `receiver_seed44/conditioning_mismatch_comparison.{json,tsv}` and `receiver_seed45/conditioning_mismatch_comparison.{json,tsv}`.

### 2026-09-22 - PathVQA frozen same-seed Static P20 stabilization control (stopped negative route)

- Exact suite target: `pathvqa_qdpt_frozen_static_p20_sandwich_seeds44_45`; implementation commit `a8219a6`. Controlled change from final Dense D768 Sandwich QDPT: initialize the LLM-side P20 from the matching seed's independently trained Static Prompt epoch3 checkpoint and freeze only P20, while reinitializing and training Visual18, A_t10 and the complete question-conditioned dynamic branch under the original PathVQA three-epoch/data-seed42 protocol. Trainable parameters are **7,753,984** instead of7,805,184. No Test evaluation was run.
- Seed45 completed as `pathvqa_qdpt_d768_question_q10_l17_p20_staticinit_frozen_s8_av10_sandwich_seed45_20260922`. Complete Validation6,259 /832 image clusters is **55.0248 Overall /89.8880 Yes-No /20.2616 Free-form**, image-clustered95% CI **[53.56,56.38]**. Per question type: how6.9767, other7.1429, what18.3673, when0.0000, where38.1418, why4.7619 and yes/no89.8880. TTFT mean0.087140s, TPOT0.034715s/token (28.806 tokens/s), request latency mean0.174061s.
- Against the original jointly trained QDPT seed45 result57.2935/90.6880/23.9949, freezing the pretrained P20 loses **2.2687 Overall**,0.8000 Yes/No and3.7333 Free-form; `where` falls15.4034 points. It is also effectively tied with the standalone Static P20 seed45 Overall55.0567 (**-0.0319**), so the retrained visual/dynamic pathway recovers no aggregate QDPT gain under this intervention.
- Seed45 output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_staticinit_frozen_s8_av10_sandwich_seed45_20260922`; predictions and summary are under `eval_validation/epoch_3`.
- Seed44 did not begin training and has **no score**. Exact attempted experiment: `pathvqa_qdpt_d768_question_q10_l17_p20_staticinit_frozen_s8_av10_sandwich_seed44_20260922`. The older standalone Static P20 seed44 checkpoint stores no seed metadata, so the strict loader rejected it with `ValueError: Static Prompt seed must match the QDPT model seed: checkpoint=None model=44`. This is a checkpoint-schema compatibility failure, not model divergence. Partial output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_staticinit_frozen_s8_av10_sandwich_seed44_20260922`; `train.log` only.
- Decision: **close this stabilization route without repairing the legacy-loader compatibility and without rerunning seed44 or adding seed46**. One completed seed cannot estimate variance, so this experiment does not prove whether freezing P20 would reduce multi-seed standard deviation. It is nevertheless sufficient for the predeclared performance gate: the2.2687-point loss greatly exceeds the allowed0.30-point mean degradation, and the result collapses to the Static Prompt baseline. The evidence rejects treating P20 as a separately pretrained, frozen stable base; QDPT's gain requires joint adaptation among P20, A_t and the conditional visual/text pathway.

### 2026-09-22 - PathVQA Dense Sandwich dynamic-branch late start seeds45/44 (stable but lower route rejected)

- Exact target: `pathvqa_qdpt_dynamic_late_start_sandwich_seeds45_44`; implementation commits `42e1676` and test-fixture correction `2a0014d`. Both runs use complete PathVQA train and Validation6,259 /832 image clusters, model seeds45 then44, data seed42,3 epochs and epoch3-only Validation. Structure, initialization, Prompt counts/order, base learning rates and the global3% warmup plus linear-decay scheduler match final Dense D768 Sandwich QDPT; no pretrained Static P20 was loaded and no Test evaluation was run.
- Controlled change: write the text dynamic Prompt as `D=A_t+g(t)Delta_theta(I,q)`. For the first10% of optimizer steps (`ceil(0.10*1845)=185`) use `g=0`; P20, A_t10 and Visual18 continue training, while the **7,709,952** dynamic-generator parameters have gradients set to `None` before AdamW so parameters, optimizer state and weight decay do not advance. At zero-based step185 set `g=1` and resume the original joint update without restarting the scheduler or adding steps.
- Seed45 exact experiment `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_dynamic_late_start10_seed45_20260922`: **57.8048 Overall**, versus original same-seed57.2935, a **+0.5113** recovery. The suite reports two-seed Free-form mean25.1595; together with seed44's integer-consistent24.5692 this implies seed45 **25.7498 Free-form**, approximately+1.7549 over the original23.9949. Its Yes/No is mathematically reconstructed as89.9520 from the reported Overall/Free-form and known3125/3134 split denominators, approximately-0.7360 from the original90.6880. Direct seed45 CI, per-question-type rows, timing and summary printout were not supplied and remain pending extraction. Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_dynamic_late_start10_seed45_20260922`.
- Seed44 exact experiment `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_dynamic_late_start10_seed44_20260922`: **57.2136 Overall /89.9520 Yes-No /24.5692 Free-form**, clustered95% CI **[55.71,58.61]**. Per type: how6.9767, other7.1429, what19.9372, when0, where61.3692, why4.7619 and yes/no89.9520. Against original same-seed60.7765/92.7360/28.9087, changes are **-3.5629 Overall /-2.7840 Yes-No /-4.3395 Free-form**; `where` falls14.6699 points. TTFT mean0.086967s, TPOT0.034364s/token (29.100 tokens/s), request latency mean0.171623s. Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_dynamic_late_start10_seed44_20260922`.
- Two-seed Overall changes from original to late-start are seed44 **-3.5629** and seed45 **+0.5113**. The mean falls from **59.0350 to57.5092 (-1.5258)**, while the absolute seed gap contracts from3.4830 to **0.5912**, an83.03% reduction. Free-form mean falls from26.4518 to **25.1595 (-1.2923)**. Both late-start seeds have the same89.9520 Yes/No point estimate; the remaining gap is concentrated in Free-form.
- Pre-registered decision: seed45 passed its preliminary Overall/Free-form gate and genuinely improved, so seed44 was correctly run. The two-seed route nevertheless fails the required mean58.7350 and Free-form mean25.9518 despite passing the maximum1.50 gap. **Reject the10% late-start intervention and do not run seed46, scan the delay ratio, add regularization, or evaluate Test.** The smaller variance is produced mainly by destroying the strong seed44 trajectory, exactly the prohibited failure mode.
- Mechanistic interpretation: delaying the dynamic branch materially changes which basin each seed reaches and can help the weak seed45, so simultaneous static/dynamic startup plausibly contributes to initialization sensitivity. It is not the sole cause and a fixed10% delay is not a usable stabilization solution: it removes3.56 points from seed44 and1.53 points from the two-seed mean. Boundary audit values from `dynamic_late_start_audit.jsonl` (parameter hash equality, zero activation residual, gradients, loss and attention entropy) have not yet been supplied; implementation fail-fast checks had to pass for both runs to reach evaluation, but exact diagnostic values remain pending extraction.

### 2026-09-23 - PathVQA V0 visual selection and question offset seed44

- Exact experiment: `pathvqa_v0_visual_selection_offset_seed44`; implementation commits `87aa5dc` and `b3c7c7a`. Complete official PathVQA Validation **6,259 questions /832 image clusters**, model seed44, data seed42, three training epochs, fixed epoch3 checkpoint and one Validation evaluation. No Test or other seed was run.
- Controlled new method: frozen Qwen3-VL backbone with trainable embedding-row P20 and code-index17 Visual18 (`S8+A_v10`), true-question-only width128 residual depthwise-k3/pointwise context and attention pooling, independent code-index5/11/17 Q/K maps, three summaries over the **same native post-merger Value**, question-conditioned layer weights and three64-wide blocks, then a token-shared width192 ADePT-style residual on real question embeddings. The legacy QDPT dynamic generator and its Z10 tokens are absent. Offset output weights initialize `Normal(0,1e-4)` with zero bias. The completed run passed the launcher preflight and the model's per-group fail-fast parameter check: **2,356,675 trainable parameters** (P20 51,200; Visual18 18,432; question context 345,088; maps 448,896; layer condition 507,267; offset 985,792).
- Validation: **55.6479 Overall** (displayed 55.65), **89.6640 Yes/No** (displayed 89.66) and displayed **21.73 Free-form**. Per question type: how9.3023, other7.1429, what17.3077, when0, where55.5012, why0 and yes/no89.6640. Image-clustered95% CI **[54.17,57.08]**. Timing: TTFT mean0.048381s (p500.047642, p950.051076), TPOT0.020096s/token (**49.76 tokens/s**), request latency mean0.099247s (p500.086741, p950.149540).
- Same-seed, same-split/epoch3 descriptive comparisons: versus Static P20 **54.8650**, V0 is **+0.7829 Overall** with about46.0x its trainable parameters; versus CoCoOp-style P20/H160 **57.4053**, **-1.7574** with about2.70x its parameters; versus Full-Attention LoRA-r8 **59.3386**, **-3.6907**; versus final Dense Sandwich QDPT **60.7765**, **-5.1286** while using about30.19% as many trainable parameters. No paired significance test has been calculated for these comparisons; overlapping standalone CIs are not a paired test. One seed cannot establish stability.
- Diagnostic files expected at the output root are `v0_first_batch_audit.json`, `v0_diagnostics.jsonl`, `train_report.json` and `preflight_tests.log`; their exact offset/question RMS, branch gradient, map entropy/layer-weight trajectories, loss and training-time values were not supplied in this report and remain pending extraction. Thus the completed evaluation establishes that V0 is trainable and beats Static P20 modestly, but does **not** yet establish a competitive parameter-performance advantage over the stronger lightweight CoCoOp-style control, nor explain the deficit's mechanism.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_offset/pathvqa_v0_visual_selection_offset_seed44_20260923`; checkpoint `checkpoints/epoch_3`; predictions and summary under `eval_validation/epoch_3`; selected score in `selected_result.tsv`.

### 2026-09-23 - V0 seed44 diagnostic supplement supplied by user

- Supplements the completed `pathvqa_v0_visual_selection_offset_seed44` run above; same PathVQA Validation, seed44/data seed42, configuration and output root. No new training or evaluation was performed in this task. Overall remains55.6479, Yes/No89.6640, displayed Free-form21.73. Source is the user's pasted audit/trajectory extraction; raw server JSON files were not independently fetched. This supersedes the preceding statement that these diagnostic values had not been supplied, without changing the historical record.
- Parameter audit confirms2,356,675 total: p20 51,200; visual_s8 8,192; visual_av10 10,240; question_context 345,088; maps448,896; layer_condition507,267; offset985,792. Preflight: Visual18 Normal(0,0.02), measured S8/Av10 std0.01988239/0.02008527, orderS8_then_Av10 at code layer17; output Normal(0,1e-4), measured std0.00009989, zero bias.
- Training completed3 epochs in6,164.6126s (about1.7124h),9.565 reported samples/s,0.299 steps/s, train_loss12.686219504338293. This is not a matched throughput comparison to the historical controlled LoRA/QDPT benchmark. Trajectory extraction contains93 diagnostic records.
- First real-batch forward: question_rms0.01982339, condition_rms0.19342539, down_rms0.01612866, native_value_rms0.72621554, offset_rms0.0001746317, offset/question0.00880937,16 question tokens. Layer5/11/17 map normalized entropy0.9998763/0.9999024/0.9999017; top mass0.00499862/0.00505552/0.00479446; layer weights0.2854624/0.3653794/0.3491583; summary RMS0.4780350/0.4780597/0.4780256. Similar RMS does not establish identical summary directions.
- First-batch gradient-group norms: p20 13.41945, S8 0.1531375, Av10 0.1548467, question_context0.002940716, maps0.000687902, layer_condition0.1954740, offset24.147509. Query/key/value-condition projection weight gradients are nonzero: layers5/11/17 query0.0001497562/0.00008223893/0.00005531780, key0.0005553048/0.0002663352/0.0002047185, value_blocks0.1098669/0.1324970/0.09251325. These value_blocks are condition projections, not evidence of a newly trained full CA Value path. First-batch audit versus trajectory gradient aggregation/clipping conventions were not verified; do not directly compare their norm magnitudes as identical measurements.
- Last10-record means: offset/question0.681297, condition_rms0.594963, down_rms0.037482, offset_rms0.0143537. Final record respectively0.56328/0.633216/0.0370365/0.0116841. Ratio of reported mean condition RMS to mean down RMS is about15.87; it is not a measured per-token ratio distribution.
- Layer5/11/17 last10 mean map entropies0.980851/0.907977/0.811040; final0.991176/0.917226/0.725843. Last10 mean top masses0.00961219/0.020403/0.061650; final0.0114216/0.0193202/0.10642. Last10 mean layer weights0.167730/0.549122/0.283148; final0.167234/0.549926/0.282840. Maps became nonuniform, especially layer17, but these values do not prove question dependence or correct localization; weight magnitudes alone do not establish causal layer importance.
- Last10 mean gradient norms: p20 0.00878564, S8 0.00388075, Av10 0.00625056, question_context0.0130757, maps0.0311971, layer_condition0.317571, offset0.944007. Final respectively0.00916384/0.00338909/0.00416129/0.0138296/0.0230235/0.299718/0.953591. The supplied statistics show active finite branches, not numerical divergence; no new implementation audit was performed.
- Diagnostic hypothesis, not a causal conclusion: in delta_i=ReLU(t_i+c+b1)W_up+b2, much larger c than t_i may make activation patterns similar across question tokens, so the visual effect behaves mainly as a shared additive offset rather than token-specific conditional rewriting. Need within-question activation/offset direction measurements and paired inference interventions to distinguish this from useful conditioning, excessive overall offset, or other bottlenecks. RMS0.68 alone is not proof that question semantics were damaged. No rescue training has been selected or authorized.

### 2026-09-23 - PathVQA V0 seed44 epoch3 read-only diagnostic first launch (import failure)

- Exact target: `pathvqa_v0_seed44_epoch3_diagnostic`; implementation revisions `97b3959` and `59c2fd1`. Dataset was intended to be complete PathVQA Validation with the existing V0 model seed44/data seed42, fixed epoch3 checkpoint, original generation protocol and original baseline predictions. Controlled change was read-only forward instrumentation plus planned `offset_off` and `condition_off` inference interventions; **no training, checkpoint edits, Test or other seed**.
- Actual result: script stopped at import time with `ModuleNotFoundError: No module named 'train.data_pipeline'; 'train' is not a package`. **No model or data were loaded, no forward probe or Validation intervention ran, and no Overall/Yes-No/Free-form/where score or paired CI exists for this attempt.** The three subsequent `QWen3WithMMRL.forward` docstring messages are not the Python exception that stopped the run.
- Diagnostic cause: the script used `from train.data_pipeline import build_target_supervision_masks`, whereas the repository's `pathvqa.data_pipeline` exposes that function after adding its `train/` directory to the import path. Corrected the import without changing any V0 model parameters or inference formula. The launcher now prints its chosen output path before running the script.
- Output: the failed launcher created a fresh directory under `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_offset/diagnostics/` containing `diagnostic.log`; its exact dated/suffixed directory name was **not printed by that launcher version** and is pending user confirmation. The original V0 output remains `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_offset/pathvqa_v0_visual_selection_offset_seed44_20260923` and is unchanged. This is an infrastructure/code-entry failure, not evidence for or against V0's mechanism.

### 2026-09-23 - PathVQA V0 seed44 epoch3 read-only diagnostic stopped by train/prefill mask mismatch

- Exact target: `pathvqa_v0_seed44_epoch3_diagnostic`; import-fix revision `b453724`. The intended scope remained the existing V0 model seed44/data seed42 epoch3 checkpoint, deterministic 128-sample Validation forward probe, followed only if the implementation audit passed by full-Validation `offset_off` and `condition_off` inference interventions. No training, checkpoint modification, Test evaluation or other seed was authorized or performed.
- The audit loaded the frozen base and V0 checkpoint successfully and re-confirmed the exact **2,356,675** trainable-tensor checkpoint contract, then stopped on the first audited mismatch at `pathvqa:validation:756`: training real-question IDs were `[12555,1558,419,2168,1473,30]`, while prefill IDs were `[12555,1558,419,2168,1473]`. Token ID30 is the final question-mark token in this case. The predeclared fail-fast condition therefore worked: **no 128-sample statistics and neither complete Validation intervention ran**, so there are no new Overall, Yes/No, Free-form, where, paired-exclusive counts or confidence intervals.
- Cause is in the existing inference fallback of `locate_question_mask`: when raw-question IDs do not exactly match the question followed by the evaluation instruction, it retains only prompt tokens whose complete character offset lies within the raw question. A boundary token spanning the question/newline boundary can therefore be dropped. Training uses the answer-bearing template and exact raw-question span, so the final punctuation is retained. This is a confirmed train/prefill conditioning-mask inconsistency, not evidence for the hypothesized common-offset degeneration.
- Consequence boundary: the image and complete textual prompt still enter the frozen base model; the mismatch specifically changes V0's question-conditioned summary/maps and which real-question embeddings receive the learned offset. The epoch3 checkpoint is not shown to be corrupt, but the reported55.6479 baseline was evaluated with the old inference-mask behavior and is now marked as that historical implementation result rather than a clean evaluation of the intended train/prefill-identical formula. The frequency and score effect are unknown because the audit correctly stopped at the first mismatch.
- Decision: stop this diagnostic route at the implementation audit as preregistered. Do not interpret `offset_off`/`condition_off`, do not choose Norm/gate/LR changes, and do not retrain yet. The smallest justified follow-up, only after separate authorization, is to repair the inference boundary alignment and rerun the **normal** epoch3 Validation baseline with the same checkpoint before deciding whether the two interventions or any retraining remain meaningful.
- Exact output: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_offset/diagnostics/pathvqa_v0_seed44_epoch3_diagnostic_20260923_1`; `diagnostic.log` contains the fail-fast traceback. Original checkpoint and predictions were not overwritten.
- Follow-up implementation (not an experiment result): the inference fallback now assigns every tokenizer token overlapping the raw-question character interval to the question, including a final token that also spans the following newline, and then requires the resulting token-ID sequence to equal the standalone training question IDs exactly. A new target, `pathvqa_v0_seed44_mask_fixed_validation`, is limited to the corrected **normal** full Validation pass with the same checkpoint and a paired comparison against the historical predictions. Its score is pending; no intervention or retraining is implied by this code change.

### 2026-09-23 - PathVQA V0 seed44 mask-fix v1 normal Validation stopped during warmup

- Exact target: `pathvqa_v0_seed44_mask_fixed_validation`; mask-fix v1 revision `ef1a0f5`. Intended dataset/protocol was complete PathVQA Validation using the unchanged V0 seed44/data seed42 epoch3 checkpoint and original generation settings, followed by a paired comparison with the historical55.6479 predictions. No training, checkpoint edit, Test, other seed, `offset_off` or `condition_off` was run.
- Actual result: the model and checkpoint loaded and the2,356,675 parameter contract passed, but the evaluation stopped in the timing warmup before any scored prediction. For the warmup question, standalone/training IDs ended in token30 while the complete prefill prompt represented the final question mark plus following newline as token5267: training `[12555,614,5558,862,96092,30]`, prefill-overlap `[12555,614,5558,862,96092,5267]`. Therefore **no Overall, Yes/No, Free-form, where, paired-exclusive counts or CI exists** for v1.
- Diagnosis: v1 correctly restored the missing boundary position but made an invalid stronger assumption that contextual tokenization inside `question + newline + instruction` must preserve the standalone final token ID. It need not; position alignment and conditioning-source identity are separate concerns. This failure is from the repair guard, not from the checkpoint or the common-offset hypothesis.
- Corrective implementation (not yet a result): v2 supplies the conditional branch with separately tokenized raw-question IDs, exactly matching training semantics, while using the overlapping full-prefill positions only as delta write targets. Source and target token counts must match one-to-one; otherwise inference fails. The complete text prompt, frozen base path and decoding protocol are unchanged. The next authorized run remains only the corrected normal Validation baseline.
- Output root is expected from this target's first allocation at `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_offset/diagnostics/pathvqa_v0_seed44_mask_fixed_validation_20260923`; the user-supplied excerpt omitted the launcher's output-path line. `mask_fixed_validation.log` contains the traceback. If the launcher printed a suffix, append a correction rather than replacing this record.

### 2026-09-23 - V0 seed44 mask-fix v2 normal Validation completed

- Target: `pathvqa_v0_seed44_mask_fixed_validation`; exact mask policy `standalone_training_source_with_prefill_overlap_targets_v2`. Same original V0 seed44/data seed42 epoch3 checkpoint, PathVQA Validation6,259 questions/832 image clusters, original generation protocol. Only normal inference after boundary/source-mask correction; no retraining, checkpoint changes, Test, other seed or offset/condition-off intervention. Source: user-supplied evaluation and paired JSON; raw server artifacts not independently fetched in this task; exact executing commit not supplied.
- Overall **56.7503**, Yes/No **89.5040**, Free-form **24.0906**. Types: how7.7519, other7.1429, what19.6625, when0, where59.4132, why0, yes/no89.5040. Standalone image-clustered95% CI[55.2321,58.2468],2,000 iterations,bootstrap seed42. Trainable parameters unchanged2,356,675.
- Paired against historical old-mask55.647867: Overall **+1.1024125**, variant-only/baseline-only correct202/133 (net69), exact McNemar p=0.0001935816, image-clustered paired95% delta CI[+0.4775435,+1.7332598],10,000 iterations,seed42. Free-form **+2.3611997**,164/90,p=3.9998786e-6; Yes/No **-0.1600**,38/43,p=0.6569925. What **+2.3547881**,130/70,p=2.6528756e-5; where **+3.9119804**,34/18,p=0.0364834 (unadjusted subgroup test); how-1.5503876,0/2,p=0.5; other/when/why unchanged.
- Correction: use56.7503 as the current normal V0 evaluation under the repaired policy; retain55.6479 as historical old-mask result. Descriptively, updated V0 is+1.8853 over Static P20 seed44 54.8650 and-0.6550 below CoCoOp-style seed44 57.4053. These baseline comparisons have no new paired tests. Improvement establishes a material inference-interface effect, not proof of correct visual localization, successful token-specific conditioning or multi-seed stability. The public-offset hypothesis remains untested. Source/target count equality alone is not proof of every contextual token's semantic alignment; v2 intentionally separates standalone conditioning source from prefill overlap write targets.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_offset/diagnostics/pathvqa_v0_seed44_mask_fixed_validation_20260923_1`. This completes v2 and supersedes preceding pending-score status. Timing and forward diagnostics were not included in this result excerpt.
- Diagnostic continuation authorized after this completion: retain the exact v2 mask/source policy and use `mask_fixed_normal` under this output root as the paired normal baseline. The 128-question probe and two interventions are pending server execution; no normal baseline rerun or rescue training is scheduled. Original training-audit files are read from the original checkpoint's training root, not from the repaired evaluation directory.

### 2026-09-23 - V0 seed44 repaired-mask offset_off and condition_off Validation interventions

- User supplied completed results for variants `offset_off` and `condition_off` of the planned `pathvqa_v0_seed44_epoch3_diagnostic` continuation. Existing V0 seed44/data seed42 epoch3 checkpoint, full PathVQA Validation6,259 questions, repaired v2 mask/source protocol; no retraining, Test or other seed. Normal paired reference is56.7503/89.5040/24.0906/59.4132 (Overall/Yes-No/Free-form/where), from `pathvqa/outputs/visual_selection_offset/diagnostics/pathvqa_v0_seed44_mask_fixed_validation_20260923_1`. Exact intervention output directory, executing commit, raw reports, CI resampling settings and128-sample probe statistics were not included in the user's excerpt and remain pending; do not assume intervention files reside in the normal-baseline directory. Scores below are copied from the user, not independently recomputed.
- `offset_off`: only final question residual delta disabled. Overall49.2251, delta-7.5252, variant-only/normal-only correct143/614, paired95% delta CI[-8.4364,-6.5907]; Yes-No84.6080, delta-4.8960,70/223,CI[-5.9360,-3.8952]; Free-form13.9438, delta-10.1468,73/391,CI[-11.6815,-8.6246]; where37.8973, delta-21.5159,11/99,CI[-26.1614,-17.0732]. Remaining parameters retained.
- `condition_off`: only visual condition c at the offset MLP addition disabled; trained text offset path retained. Overall51.9412, delta-4.8091,123/424,CI[-5.5663,-4.0373]; Yes-No86.6560, delta-2.8480,63/152,CI[-3.7544,-1.9500]; Free-form17.3261, delta-6.7645,60/272,CI[-8.0287,-5.5752]; where42.7873, delta-16.6259,6/74,CI[-20.6825,-12.7139].
- Both interventions substantially damage the existing model, especially open/spatial questions. The current checkpoint depends on both the offset path and its visual condition; zeroing the condition is not a rescue. This does not measure retrained standalone component contributions, prove sample-specific visual grounding or exclude harmful common-offset behavior. Zeroing c changes the ReLU regime and inference distribution; a shared learned bias can also be important. The2.7161-point difference between intervention scores is descriptive, not an additive decomposition of text/vision contributions. No normalization, LR, width, retraining or further inference sweep selected from these results. Await existing128-sample probe output before choosing one rescue hypothesis.

### 2026-09-24 - V1 seed44 first training launch stopped at optimizer step 49

- Exact experiment/target: `pathvqa_v1_visual_selection_prefix_p20_seed44`, executing implementation commit `807cc9b`; PathVQA train, model seed44/data seed42, intended 3 epochs and fixed epoch3 Validation. Controlled V1 change relative to V0: preserve three-layer question-guided visual selection, common native Value and Visual18; replace question-token residual with one shared conditional shift on existing P20. Expected/audited trainable count: 1,864,963. Source: user-supplied server traceback and diagnostics; this run did not reach a checkpoint or Validation, so Overall, Yes/No, Free-form and per-type scores are **not available**. Test/other seeds were not run.
- Progress/failure: at displayed `49/1845` optimizer steps after about 4m49s, `condition_p20` raised `RuntimeError: V1 shared shift exceeds bf16 tolerance: 0.01171875`. The first real-batch audit had passed. Available step40 diagnostics: finite gradients for all seven groups; `condition_rms=0.25596842`, `offset_rms=0.00543662`, `p20_rms=0.63685185`, `offset_to_p20_rms=0.00853672`; normalized map entropies for layers5/11/17 were `0.99983507/0.99989104/0.99987507` approximately. The launcher then reported one failed experiment.
- Cause/correction: the code broadcasts one identical float shift to all20 P20 positions, but the guard recovered an "effective delta" by subtracting distinct **BF16-rounded** base P20 values after addition. As P20 magnitudes grew with LR0.3, position-dependent BF16 rounding produced a spread over the arbitrary0.01 threshold. This is a false-positive monitoring assertion; it does not establish an architecture failure or a change in the intended shared-shift formula. Follow-up code removes only the invalid numeric gate, retains finite/shape and native-embedding checks, and logs the BF16 effective-delta spread as a descriptive diagnostic. Forward computation, optimizer and schedule are unchanged. No intermediate checkpoint exists (`save_strategy=no`), so a rerun must start from step0.
- Output path (confirmed from the user's `[PATHVQA_V1_CONFIG]` line): `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_seed44_20260924`; `train.log` and `v1_real_batch_preflight.json` are the expected partial artifacts. Do not treat this partial run as a scored V1 result.

### 2026-09-24 - V1 conditional prefix P20 seed44 completed (accumulation normalization deviation)

- Exact experiment: `pathvqa_v1_visual_selection_prefix_p20_seed44`; PathVQA, model seed44/data seed42, 3 epochs, fixed epoch3 complete Validation (6,259 questions, 832 image clusters); no Test or other seeds. Controlled change from V0: retain question-guided layer5/11/17 selection, common native Value, condition aggregation and Visual18; replace real-question offsets with one shared Linear(192,2560)(ReLU(c)) shift on existing P20 before full chat. Trainable parameters 1,864,963. Plan records running commit `4636416ee99768c667377f0c66b5809daf48ffcd`; final evaluation commit not independently verified. Source: user-supplied completion/paired output, not independently retrieved server artifacts.
- Overall **57.5491**; Yes/No **89.4080**; Free-form **25.78** (reported rounded). Types: how9.3023, other7.1429, what21.0754, when0, where62.8362, why4.7619, yes/no89.4080. Image-clustered95% CI [56.03,58.96],832 clusters; standalone bootstrap settings not supplied.
- Paired Overall, variant-only/baseline-only correct: repaired V0 56.7503 ->57.5491, delta+0.7988,316/266,95% clustered delta CI[-0.06516091499313523,1.6414386027537595]; CoCoOp-style seed44 57.4053 ->57.5491,delta+0.1438,318/309,CI[-0.6943600578013559,0.9793063377950703]; Static P20 seed44 54.8650 ->57.5491,delta+2.6841,422/254,CI[1.7787777063745518,3.572553099204359]. Each paired bootstrap uses10,000 iterations,seed42,832 image clusters. No paired subgroup CIs supplied.
- Timing: TTFT count6259,mean0.083398,p500.083064,p950.087939,min0.061458,max0.38804 seconds; TPOT0.035026 seconds/token,28.55 tokens/sec; request mean0.166382,p500.141294,p950.253236,min0.102527,max0.696552 seconds. Final training runtime/peak memory/final diagnostics not supplied. Historical timing comparisons are not controlled hardware/runtime benchmarks.
- Protocol deviation, reported by user: Transformers5.0.0/Accelerate1.12.0; direct real-batch forward loss3.8914. Wrapper forward(**kwargs) causes Trainer model_accepts_loss_kwargs=True, but backbone loss ignores window num_items_in_batch; Trainer skips division by16 and Accelerate accumulation is1. Logged loss is approximately a sum of16 microbatch means; pre-clipping gradients are approximately16x the equal-microbatch mean gradient. Post-clipping group norms versus pre-clipping Trainer grad_norm explain their different scales. AdamW/clipping preclude claiming16x parameter updates. This run completed unchanged and must retain this deviation label. Same-logits manual CE and independent full/partial accumulation-window numerical checks remain pending; historical baseline runtime/version applicability is unknown.
- Interpretation: positive single-seed gain over Static with paired CI excluding0; positive point estimate versus V0 but CI includes0; no established gain or equivalence versus CoCoOp. Where rises3.4230 versus repaired V0 and5.1345 versus CoCoOp, descriptive only and not proof of localization. V1 uses about2.14x CoCoOp parameters and about20.86% fewer than V0. No stability conclusion or clean injection-location causal attribution (output head also changed). Retain checkpoint; complete normalization and historical-protocol audit before selecting corrected retraining.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_seed44_20260924_1`; checkpoint `checkpoints/epoch_3`; predictions `eval_validation/epoch_3/pathvqa_predictions.json`; summary `eval_validation/epoch_3/pathvqa_summary.json`.

### 2026-09-24 - V1 seed44 epoch3 loss/gradient accumulation read-only numerical audit

- Exact diagnostic target: `pathvqa_v1_loss_scaling_audit`; implementation commit `e42ae46`, executing Git hash is recorded by `v1_loss_scaling_audit.json` but was not present in the user-pasted terminal excerpt. Dataset: PathVQA train19,654, data seed42, model seed44, fixed completed V1 epoch3 checkpoint from `pathvqa_v1_visual_selection_prefix_p20_seed44_20260924_1` (training commit `4636416ee99768c667377f0c66b5809daf48ffcd`). Controlled diagnostic change: at fixed parameters, compare hand-scaled equal-microbatch CE gradient, original `V1Trainer.training_step` and the candidate with only `trainer.model_accepts_loss_kwargs=False`; no optimizer/scheduler step, parameter update, real clipping, retraining, Validation or Test. Thus **no new Overall/Yes-No/Free-form scores**; the existing V1 Validation **57.5491/89.4080/25.78** remains unchanged and retains its normalization-deviation label. Independent output: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_loss_scaling_audit_20260924` (`v1_loss_scaling_audit.json`, `audit_summary.md`, adjacent launcher log).
- Same-logits causal CE: after P20 insertion,12 effective target tokens, FP32 shifted CE sum20.23833656 and mean **1.6865280867**; direct `outputs.loss` **1.6865280867**, absolute/relative error **0**. This is the trained epoch3 checkpoint, not the initial real-batch preflight loss3.89144063. Supplying full-window `num_items_in_batch=228` to the model leaves the loss exactly1.6865280867 (difference0), confirming that the count does not enter the V1 backbone loss denominator. Supervised examples decode to newline + `an abnormal tripolar spindle` / `no` + chat end/newline, consistent with answer supervision rather than P20 targets; the first20 expanded labels are all `-100`.
- Prepared Trainer DataLoader used batch2 on19,654 rows, giving9,827 microbatches and an actual **3-microbatch tail window**. The audited sampler order is a fresh equivalent sampler, not a replay of the historical epoch3 RNG order. Full16-window input has228 nonignored labels; tail3 has50. Trainer passes `num_items_in_batch=228/50` on the original path and none on the candidate path. Original loss is returned un-divided, candidate `training_step` returns each microbatch loss divided by the actual window size16/3.
- Full K16, **overall preclip**: hand reference norm0.96141398, original Trainer15.37775517 (**15.9949361×**, cosine0.9999924, relative L2 error14.99498), corrected candidate0.96152389 (**1.0001143×**, cosine approximately1, relative L2 error0.0054861). All seven trainable groups show original/reference norm ratios approximately15.99–16.03 and corrected/reference ratios approximately1.0001–1.0026. Candidate passes the preregistered numeric check.
- Tail K3, **overall preclip**: hand reference norm1.78812253, original Trainer5.36089230 (**2.9980564×**, cosine0.9997904, relative L2 error1.99844), corrected candidate1.78791976 (**0.9998866×**, cosine approximately1, relative L2 error0.0079305). Group ratios vary modestly with BF16 accumulation (original about2.959–3.065; corrected about0.989–1.0013), but the overall result confirms division by the actual3 rather than a fixed16. Both window checks report `true`; matched RNG and zeroed gradients were used for each path. Cosines slightly above1 in a few float reductions are numerical roundoff, not stronger-than-perfect alignment.
- Threshold1 global clipping on **gradient copies**: K16 hand reference0.961414, original1.000001, candidate0.961524; K3 hand/reference and both compared copies are approximately1.0. Clipping thus removes most of the original norm inflation and prevents interpreting preclip16×/3× as corresponding parameter-update multipliers; it does not undo the protocol deviation, especially for windows/groups below clipping and optimizer dynamics. No actual diagnostic gradient was clipped or applied. The script passed its parameter SHA256 before/after check and exited `[DONE]`.
- Audit interpretation: **forward token-mean CE is correct; the original Trainer/Accelerate accumulation path is wrong for the intended equal-microbatch mean because the model ignores `num_items_in_batch`; setting only `model_accepts_loss_kwargs=False` numerically matches that intended objective for K16 and K3.** Do not additionally divide loss or alter Accelerate accumulation; do not silently reinterpret the target as whole-window token-weighted CE. This confirms the V1 gradient-scaling defect, not that the existing Validation score should be arithmetically corrected or that all earlier methods share it. The user excerpt omits the JSON's exact runtime package versions, source locations, parameter hashes, and five-method historical provenance table; previously reported server versions are Torch2.8.0+cu128, Transformers5.0.0, Accelerate1.12.0, but full audit evidence and historical classifications remain pending extraction. The independently guarded corrected-training entry exists but was **not** run; any matched seed44 corrected rerun requires separate authorization.
- Supplement from the user-provided `audit_summary.md`: executing audit commit is **`e42ae4608bcbb93137b3c9e410a565f03800237e`**, and runtime is explicitly **PyTorch2.8.0+cu128 / Transformers5.0.0 / Accelerate1.12.0**. Trainer `model_accepts_loss_kwargs=True`, Trainer accumulation16, Accelerate accumulation1; installed loop clips before `on_pre_optimizer_step` and calls optimizer afterward, while this audit intercepted before clipping. The summary confirms no optimizer/scheduler step or checkpoint write. This corrects the preceding 'runtime package versions omitted' status without altering any result.
- The summary's historical seed44 provenance table gives **unknown run commit and unknown recorded runtime versions for all five**: V0 `pathvqa_v0_visual_selection_offset_seed44_20260923`, CoCoOp `pathvqa_cocoop_style_p20_h160_seed44_20260909`, Static P20 `pathvqa_prompt_tuning_len20_seed44_20260827`, final QDPT Sandwich `pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260909`, and LoRA-r8 `pathvqa_lora_full_model_attention_r8_seed44_20260830`. Their loss-kwargs support, `num_items_in_batch` denominator path, and Trainer/Accelerate normalization during those historical runs remain **evidence insufficient**, not confirmed affected or unaffected. The table's repeated 'PEFT wrapper or source unavailable' phrase is a fallback for unavailable training-commit source, **not** evidence that the four Prompt-family wrappers use PEFT; current checkout has `forward(**kwargs)` wrappers for V0/CoCoOp/Static/QDPT, while LoRA is separately wrapped by PEFT. Current source similarity cannot retrospectively establish old runtime versions or exact loss protocols. No historical score is modified.

### 2026-09-24 - V1 normalization-fixed seed44 epoch3 Validation completed

- Exact experiment: `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44`; PathVQA model seed44/data seed42, authorized from-scratch matched 3-epoch run, fixed epoch3 complete Validation6,259 questions/832 image clusters. Sole intended training change from prior V1: `trainer.model_accepts_loss_kwargs=False`, restoring equal-microbatch mean accumulation using actual full/tail window sizes. V1 architecture/initialization/LR groups/optimizer/clipping/schedule unchanged; configured trainable count1,864,963. Source: user-supplied evaluation output; executing commit, final train report, runtime/version attestation and final diagnostics not supplied in this excerpt. No new Test/other-seed result reported.
- Overall **58.57** (rounded terminal value; exact unrounded summary pending); Yes/No **90.2080** (headline90.21); Free-form **27.03** (rounded). Types: how9.3023, other7.1429, what21.9780, when0, where66.7482, why4.7619, yes/no90.2080. Image-clustered95% CI[57.07,60.05],832 clusters; standalone resampling settings not supplied.
- Descriptive changes versus original V1 normalization-deviation result57.5491/89.4080/25.78/62.8362: Overall approximately+1.02, Yes-No+0.8000, Free-form approximately+1.25, where+3.9120 percentage points. Versus repaired V0 Overall56.7503, approximately+1.82; versus CoCoOp57.4053, approximately+1.16; versus Static54.8650, approximately+3.71. No paired comparison output was supplied for this new run; do not reuse old V1 paired CIs or infer significance from standalone CIs. Historical baseline normalization protocols remain unknown.
- Timing seconds: TTFT count6259,mean0.084144,p500.083734,p950.090617,min0.061189,max0.487069; TPOT0.035596 seconds/token,28.093 tokens/sec; request count6259,mean0.169154,p500.144761,p950.256724,min0.102099,max0.746797. Train runtime/peak GPU memory/final loss and gradient/map diagnostics pending.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44_20260924`; predictions `eval_validation/epoch_3/pathvqa_predictions.json`; summary `eval_validation/epoch_3/pathvqa_summary.json`; planned checkpoint `checkpoints/epoch_3` (checkpoint line not included in this excerpt).
- Interpretation: corrected normalization gives a higher point estimate in the matched seed44 experiment, with improvements in both answer types and where; now use this explicitly named corrected run as the current V1 exploration reference. Keep old57.5491 and all historical results unchanged. One seed does not establish stability, paired significance, or a general normalization-fix benefit for other architectures. Next step is extract exact summary, paired prediction comparisons and final train metadata from existing artifacts; no new training or Test scheduled.

### 2026-09-25 - V1 normalization-fixed read-only diagnostic preflight failure

- Exact target: `pathvqa_v1_norm_fixed_seed44_diagnostic` on PathVQA Validation, model seed44/data seed42, fixed `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44_20260924/checkpoints/epoch_3`; diagnostic implementation commit `7c0e80c`. Controlled operations: read-only paired comparisons, fixed 128-question forward probe, and a small intervention preflight. No training, parameter update, Test, other seed, or full Validation intervention was completed. Therefore this run has **no new Overall/Yes-No/Free-form intervention score**; the existing corrected V1 reference remains approximately **58.57/90.2080/27.03** pending exact summary extraction.
- User-supplied traceback: during the uniform-map preflight, `diagnose_pathvqa_v1_norm_fixed.py:495` attempted `p["grid"]` and raised `KeyError: 'grid'`. The V1 hook wrote `grid` only when `diagnostic_capture_features=True`, whereas preflight used `diagnostic_probe={}` without feature capture. This is missing diagnostic metadata, not evidence that the map, checkpoint, or model forward failed. The Transformers warning that `temperature/top_p/top_k` generation flags may be ignored is separate and is not the traceback cause.
- Earlier same-run artifacts may include `normal_verified.json`, `paired_baselines.json`, `condition_question_pairs.json`, and `probe_128.json`, but their contents were not supplied and must not be interpreted yet. The exact automatically allocated output directory was not included in the supplied traceback; expected location is under `pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_norm_fixed_seed44_diagnostic_*`. Fix: always record current `grid` whenever a diagnostic probe is active, while copying full visual features only for the explicit capture probe. Rerun the same read-only target in a new directory; the small-sample gate remains mandatory.

### 2026-09-25 - Normalized V1 seed44 paired comparison and checkpoint interventions completed

- Exact diagnostic target: `pathvqa_v1_norm_fixed_seed44_diagnostic`; PathVQA complete Validation6259 questions/832 image clusters, fixed normalized V1 seed44/data seed42 epoch3 checkpoint. Source: user-pasted completed report.md; raw JSON, final executing commit and detailed preflight outputs not independently retrieved. No training, parameter updates, Test or new model seeds. This completes the retry after the historical grid-metadata preflight failure; preserve that failed-run entry.
- Normal reference exact extracted scores: **58.5717 Overall /90.2080 Yes-No /27.0262 Free-form**, what21.9780,where66.7482. This supplements earlier rounded58.57/27.03 without changing the checkpoint.
- Paired baselines (V1 minus comparator), image-cluster bootstrap10000,seed42: original V1 delta+1.0225,95%CI[+0.2861,+1.7806],V1-only280/comparator-only216; repaired V0 +1.8214,CI[+0.9927,+2.6473],355/241; CoCoOp +1.1663,CI[+0.4047,+1.9510],325/252; Static P20 +3.7067,CI[+2.7645,+4.6560],468/236. Deltas copied as reported (computed from unrounded scores), not recomputed from rounded endpoints. All four intervals exclude0; inference is conditional on these fixed checkpoints and evaluation sampling, not seed robustness or isolation of architecture from differing historical protocols. Historical comparator normalization protocols remain unverified.
- Full Validation interventions versus normal: same_type condition-question mismatch **55.6958**,delta-2.8759,95%pairedCI[-3.4472,-2.2992],normalized prediction changed fraction0.1396; cross_type **51.8933**,delta-6.6784,CI[-7.4929,-5.8937],changed0.2315; uniform_map **57.3574**,delta-1.2143,CI[-1.7321,-0.7077],changed0.1294. Intervention Yes-No/Free-form/what/where, discordant correct counts and full mismatch coverage were not included in the report excerpt and remain pending; do not infer them from totals. Conditions modify only the conditional path as designed; raw audit checks not independently inspected this turn.
- Forward probe: fixed128 IDs, same-image Q-swap128/128 coverage, same-Q condition-image swap99/128. report.md points to probe_128.json for map/summary/offset distributions and zero-norm counts but provides none of those values; therefore layer redundancy, condition-image sensitivity magnitude and information loss in aggregation remain unassessed. Question mismatch uses question-text-only types, not reference-answer-derived evaluation types. Disagreement images exported; unreviewed examples remain uncertain.
- Interpretation: both same-type and cross-type mismatches damage this checkpoint, supporting question-content sensitivity beyond the coarse question-type partition; larger cross-type loss is not an additive decomposition of type and semantic effects. Uniform pooling hurts, supporting dependence on learned nonuniform spatial selection, not correct localization or necessity of all three layers. Do not treat uniform-map score57.3574 being near historical CoCoOp57.4053 as equivalence or as a controlled architectural comparison. Next useful reading is subgroup intervention statistics and probe_128.json; no new training authorized by receipt of this report.
- Exact output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_norm_fixed_seed44_diagnostic_20260925_1`; report.md and probe_128.json. Other machine-readable artifact filenames/contents to verify on extraction.

### 2026-09-25 - Existing V1 diagnostic artifacts: subgroup, probe, and exported-image review

- Read-only follow-up of the **same completed** `pathvqa_v1_norm_fixed_seed44_diagnostic` (PathVQA Validation, model seed44/data seed42, epoch3 normalized V1 checkpoint), not a new run. Independently read existing `interventions.json`, `condition_question_pairs.json`, `probe_128.json`, `disagreements.json`, and all 60 already-exported disagreement JPGs from `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_norm_fixed_seed44_diagnostic_20260925_1` after a read-only transfer. No forward, generation, training, checkpoint write, Test, or other seed. Overall reference and interventions remain 58.5717 normal; 55.6958 same-type mismatch; 51.8933 cross-type mismatch; 57.3574 uniform map. The following are **exploratory subgroups**. Accuracy and CI units are percentage points. CI is paired image-cluster bootstrap, 10,000 resamples, seed42; `I-only/N-only` means intervention-only / normal-only correct.

| Intervention | Group (questions/images) | Normal | Intervention | Delta | I-only/N-only | Paired 95% delta CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Same-type Q mismatch | Yes/No (3125/810) | 90.2080 | 88.9600 | -1.2480 | 32/71 | [-1.8477,-0.6581] |
| Same-type Q mismatch | Free-form (3134/821) | 27.0262 | 22.5271 | -4.4990 | 50/191 | [-5.4685,-3.5334] |
| Same-type Q mismatch | what (2548/817) | 21.9780 | 17.5039 | -4.4741 | 38/152 | [-5.4667,-3.4651] |
| Same-type Q mismatch | where (409/408) | 66.7482 | 60.1467 | -6.6015 | 12/39 | [-10.0244,-3.1863] |
| Cross-type Q mismatch | Yes/No (3125/810) | 90.2080 | 87.3920 | -2.8160 | 45/133 | [-3.6141,-2.0249] |
| Cross-type Q mismatch | Free-form (3134/821) | 27.0262 | 16.4965 | -10.5297 | 50/380 | [-11.8933,-9.1783] |
| Cross-type Q mismatch | what (2548/817) | 21.9780 | 14.5604 | -7.4176 | 34/223 | [-8.7042,-6.1782] |
| Cross-type Q mismatch | where (409/408) | 66.7482 | 32.2738 | -34.4743 | 16/157 | [-39.7561,-29.0954] |
| Uniform map | Yes/No (3125/810) | 90.2080 | 89.1520 | -1.0560 | 20/53 | [-1.5894,-0.5311] |
| Uniform map | Free-form (3134/821) | 27.0262 | 25.6541 | -1.3720 | 58/101 | [-2.2538,-0.5091] |
| Uniform map | what (2548/817) | 21.9780 | 21.1146 | -0.8634 | 50/72 | [-1.7728,+0.0390] |
| Uniform map | where (409/408) | 66.7482 | 61.6137 | -5.1345 | 8/29 | [-8.0882,-2.4390] |

- `condition_question_pairs.json` reports **same-type 6259/6259 = 100% actual coverage**, as does cross-type; there are **zero unchanged questions** due to absent donors. Pairing uses question-text-only types rather than reference-answer-derived scoring types. Both mismatch variants retain the LLM-visible original question and image. The cross-type loss, especially exploratory where, does not isolate question type from changed content or prove localization. Uniform-map what CI includes zero; do not claim that particular subgroup has a confirmed loss.
- Fixed128 probe, percentiles below are **p05 / p25 / p50 / p75 / p95**. Layer pairs 5↔11, 5↔17, 11↔17 respectively: common-grid map JS = `0.185/0.239/0.303/0.358/0.443`, `0.223/0.289/0.357/0.422/0.499`, `0.106/0.144/0.221/0.286/0.402`; preprojection common-Value summary cosine = `0.863/0.924/0.951/0.973/0.985`, `0.747/0.876/0.931/0.960/0.983`, `0.851/0.927/0.965/0.983/0.993`; symmetric relative summary L2 = `0.186/0.254/0.343/0.432/0.573`, `0.191/0.309/0.389/0.554/0.817`, `0.139/0.199/0.290/0.430/0.604`. Distinct maps become more similar in common-Value summaries, but summary relative differences are still substantial; high cosine alone does not establish redundancy. 11↔17 is the closest pair by these metrics, not a demonstrated dispensable layer.
- Normal final prefix shift `b(I,Q)`: L2 norm percentiles `31.57/36.25/40.63/44.25/49.42`; shift/P20 RMS ratio `0.292/0.336/0.376/0.410/0.458`. Probe-set shift mean norm31.14; per-example mean-centered shift norm `10.20/17.38/22.50/34.13/40.28`. All128 normal shifts have defined non-near-zero norms. There is a sizable common component but also nontrivial variation; not a collapsed single shift.
- Same-image different-question probe covers128/128: raw offset symmetric relative change percentiles `0.002/0.014/0.189/1.237/1.355`, centered `0.004/0.024/0.400/1.643/1.906`; raw cosine `0.101/0.247/0.990/1.000/1.000`, centered `-0.775/-0.282/0.935/1.000/1.000`. Layer5/11/17 map-JS medians `0.0097/0.0058/0.0100`; summary-relative medians `0.0539/0.0561/0.0589`. Same-question different **condition-image features** covers99/128: raw offset relative `0.065/0.166/0.489/0.843/0.963`, centered `0.140/0.296/0.794/1.493/1.754`; raw cosine `0.537/0.649/0.889/0.988/0.999`, centered `-0.516/-0.077/0.733/0.968/0.995`. Layer map-JS medians `0.265/0.264/0.339`; summary-relative medians `0.831/0.849/0.884`. No undefined/near-zero offset norms in these comparisons. The conditional-image probe leaves the native LLM image unchanged and does not decode answers.
- On the **same 99 image-swap-eligible rows**, question-swap raw relative offset change is `0.005/0.025/0.241/1.259/1.372` and centered `0.004/0.050/0.676/1.681/1.918`; image-swap values are the 99-row series above (raw median0.489, centered median0.794). These are donor-policy-dependent forward sensitivities, not comparable causal effect sizes. The missing29 image swaps arise because the script accepts only a prior *different-image* donor with the exact same `image_grid_thw`, with a maximum16-key feature bank. Among missing rows: yes/no13, what8, where1, how6, when1; five are confirmed initial bank anchors, while the individual reason for the other24 cannot be resolved because per-row grid/bank-state was **not saved**. Do not classify them more finely or silently impute data.
- Reviewed the preselected 15 examples in each of `what_v1_only`, `what_cocoop_only`, `where_v1_only`, `where_cocoop_only` (all60 exported images, no resampling). Visible patterns: many `where` errors are broad organ/organ-system classification rather than within-image spatial relations (e.g. V1-only spleen-vs-GI IDs2826/2758/2592; CoCoOp-only oral-vs-heart/urinary IDs4601/4630, pancreas-vs-oral/urinary IDs5083/5070/5031). Some `what` disagreements are answer-granularity or scoring-language cases: hand/extremities IDs1366/1382, liver/hepatobiliary IDs3273/3578, and `miliary spread`/`miliary` ID5235; ID3299 has reference `this`, V1 `this`, CoCoOp a descriptive lesion answer. Some gross anatomy is recognizable (lung ID5207, foot/skin ID5433), but many histology images require domain expertise and are marked visually uncertain. This audit does **not** establish that learned maps select correct pathological regions or explain every paired score difference.
- Evidence-based decision: keep the question-conditioned prefix path and learned nonuniform maps for now; both have measurable checkpoint dependence. The data do **not** isolate the value of each of the three layers, and 11↔17's partial similarity is insufficient to name a safe layer to delete or a defensible parameter saving. Priority simplification question is whether all three independently learned map heads are necessary, but no specific deletion/retraining is authorized or selected. Historical comparator normalization remains unknown and one seed cannot establish stability.

### 2026-09-25 - Normalized V1 seed45 epoch3 Validation completed

- Exact experiment: `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed45`; PathVQA model seed45/data seed42, from-scratch3 epochs, fixed epoch3 complete Validation6259/832 images. Authorized controlled variable is model initialization seed44 ->45; corrected accumulation, architecture,1,864,963 trainable parameters and remaining training/evaluation settings intended unchanged. Source: user-supplied terminal results; run commit, runtime environment, final train report and diagnostics not provided here. No Test or other additional seed evaluated.
- Overall **57.17** (rounded; exact summary pending), Yes/No **88.9920**, Free-form **25.43** (rounded). Types: how8.5271,other14.2857,what20.5259,when0,where63.5697,why4.7619,yes/no88.9920. Image-clustered95%CI[55.67,58.58],832 clusters; resampling settings not included.
- Against corrected V1 seed44 58.5717/90.2080/27.0262/66.7482, Overall drops approximately1.40,Yes-No1.2160,Free-form approximately1.60,where3.1785 points. Two-seed Overall mean approximately57.87 and gap1.40; Free-form mean approximately26.23. Use exact seed45 summary before quoting more precision. Original QDPT seeds44/45 mean59.0350,gap3.4830; new gap is smaller but mean is also approximately1.16 lower, so do not claim a no-performance-cost stability fix. New parameter count is about23.89% of old7,805,184 (76.11% reduction). Two seeds do not establish a reliable variance reduction.
- Historical same-seed CoCoOp45 Overall56.3988: V1 approximately+0.77; both observed V1 seeds exceed corresponding historical CoCoOp scores in point estimate. No new paired comparison for seed45 and historical normalization protocols still unverified.
- Timing seconds: TTFT count6259,mean0.049916,p500.049611,p950.053239,min0.029662,max0.48125; TPOT0.016547,60.432 tokens/s; request count6259,mean0.093891,p500.082913,p950.137895,min0.050157,max0.549581. Large speed difference from seed44 (TTFT0.084144,TPOT0.035596) requires checking GPU/runtime/attention backend/generation and timing settings; it is not evidence of a seed-driven speedup or proof of a protocol mismatch. Training time/peak memory pending.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed45_20260925`; predictions `eval_validation/epoch_3/pathvqa_predictions.json`; summary `eval_validation/epoch_3/pathvqa_summary.json`. Keep seed44 unchanged. Reasonable next training candidate is unchanged seed46 to complete a3-seed check, not yet authorized by this result message.

### 2026-09-25 - Normalized V1 seed46 completed and three-seed exploratory summary

- Exact experiment: `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed46`; PathVQA model seed46/data seed42, intended matched3 epochs and fixed epoch3 complete Validation6259/832 images, corrected accumulation,1,864,963 parameters. Only intended variable versus seeds44/45 is initialization seed. User supplied evaluation output; executing commit, training/runtime attestation and final diagnostics not included. No Test or additional seed.
- Overall **57.58** (rounded, exact summary pending), Yes/No **90.1760**, Free-form **25.08** (rounded). Types: how7.7519,other7.1429,what20.0157,when0,where64.5477,why4.7619,yes/no90.1760. Image-clustered95%CI[56.15,58.99],832 clusters; bootstrap details omitted.
- TTFT seconds count6259,mean0.049875,p500.049747,p950.053006,min0.029815,max0.377339; TPOT0.016247seconds/token,61.548tokens/sec; request mean0.093465,p500.084911,p950.130747,min0.050871,max0.465655. Similar to seed45 timing, still unlike seed44; no controlled speedup claim. Runtime/peak training memory pending.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed46_20260925`; predictions `eval_validation/epoch_3/pathvqa_predictions.json`; summary `eval_validation/epoch_3/pathvqa_summary.json`.
- Three-seed exploratory summary using seed44 exact reported58.5717 and rounded seed45/46 57.17/57.58: Overall approximately **57.77 +/-0.72 sample SD**, range1.40. Free-form27.0262/25.43/25.08 gives approximately25.85 +/-1.04. Recompute from raw summaries before publishing precision. Historical QDPT Overall59.1522 +/-1.7528; V1 mean is approximately1.38 lower, observed SD lower, parameter count76.11% lower. This is a parameter/performance/observed-variability tradeoff, not a no-loss stability fix; three seeds do not establish universal stability and historical normalization protocols are unverified. Historical CoCoOp56.3136 +/-1.1367: V1 mean approximately1.46 higher with2.14x parameters, all3 matched-seed point estimates higher, no seed45/46 paired statistics supplied.
- Freeze normalized V1 as reference. No additional training launched. Design candidates documented separately; preserve all old results and deviation labels.

### 2026-09-25 - V2 per-position layer mixing seed44 completed (no demonstrated gain)

- Exact experiment: `pathvqa_v2_layer_mix_prefix_p20_norm_fixed_seed44`; PathVQA model seed44/data seed42, intended matched3 epochs/fixed epoch3 complete Validation6259 questions/832 images. Controlled change versus normalized V1: add60 trainable all-one-initialized alpha[20,3] coefficients to mix existing3 output components independently at each P20 position; keep one shared output bias. Configured total1,865,023 parameters, alpha LR1e-4, other settings unchanged. User-pasted final results; executing commit, final alpha/gradient diagnostics, initialization-equivalence outputs, train runtime/peak memory not included. No Test/other seeds.
- Overall **58.0764**, Yes/No **90.3040**, Free-form **25.9413**; types how10.0775,other7.1429,what21.0361,when0,where64.0587,why4.7619,yes/no90.3040. Standalone image-clustered95%CI[56.62,59.50],832 clusters; standalone bootstrap settings not supplied.
- Paired V2 minus V1, image-cluster bootstrap10000,seed42. Overall n6259/clusters832: V158.5717,delta-0.4953,V2-only217/V1-only248,CI[-1.2189475167666033,0.2228243787162268]. Yes-No n3125/clusters810: V190.2080,delta+0.0960,93/90,CI[-0.737197224895733,0.966183574879227]. Free-form n3134/clusters821: V127.0262,delta-1.0849,124/158,CI[-2.2284144420138943,0.060848073412388286]. What n2548/clusters817: V121.9780,delta-0.9419,88/112,CI[-2.1730382293762576,0.27910964132963584]. Where n409/clusters408: V166.7482,delta-2.6895,32/43,CI[-6.862745098039215,1.466992665036675]. Subgroups exploratory; all supplied intervals include0, neither established harm nor equivalence.
- Timing seconds: TTFT n6259,mean0.050350,p500.050236,p950.053433,min0.030205,max0.349640; TPOT0.016425seconds/token,60.884tokens/sec; request mean0.095147,p500.084730,p950.136819,min0.052008,max0.390622. Do not infer speedup versus different historical runtime conditions.
- Output root: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_layer_mix_prefix/pathvqa_v2_layer_mix_prefix_p20_norm_fixed_seed44_20260925`; predictions `eval_validation/epoch_3/pathvqa_predictions.json`; summary `eval_validation/epoch_3/pathvqa_summary.json`.
- Decision: keep normalized V1 as reference; no observed advantage justifies promoting this60-parameter variant or automatically adding seeds/tuning. Overall net31 fewer correct, including34 fewer Free-form and3 more Yes-No. This specific learned per-position mixing has no demonstrated gain; not proof that freedom/capacity is generally unnecessary or that V1's output is optimal. Alpha trajectories, across-position differences and effective shifts remain unexamined from this excerpt, so do not attribute the result to alpha collapse, harmful specialization or inadequate LR.

### 2026-09-25 - V2 seed44 alpha read-only checkpoint audit completed

- Target: completed V2 `pathvqa_v2_layer_mix_prefix_p20_norm_fixed_seed44`, PathVQA model seed44/data seed42 epoch3. Controlled operation: read checkpoint alpha/output weights and93 existing diagnostic records; no model forward, optimizer step, training, Validation, Test or new seed. No new accuracy; V2 remains58.0764 Overall/90.3040 Yes-No/25.9413 Free-form. Source is user-pasted audit report; no independent raw artifact retrieval this turn.
- Final alpha change-from-one RMS0.0136534,maximum absolute change0.03317368; change energy59.648% common across positions and40.352% position-varying. This40.352% is a fraction of a small parameter change, not a percentage of model output or performance contribution. Final columns5/11/17: means1.01014495/1.00657051/1.01369260; across-position std0.009125557/0.0049913431/0.010838717; minima0.99295413/0.99801952/0.99595964; maxima1.02968097/1.01748502/1.03317368. Corresponding output weight Frobenius norms1.3595457/0.75398143/1.4094597.
- W-weighted effective-mapping across-position relative difference **R=0.0094413683** (approximately0.944%). This is variation around the position-mean mapping within V2, not a V2-versus-V1 mapping difference and not measured sample-level shift difference.
- Logged steps0/920/1840 (epochs0/1.496591/2.993182): alpha-change RMS0/0.011422552/0.013653374; clipped pre-optimizer alpha gradient norms0.0000554972158665/0.0185996729031/0.0119410334620. Means step9201.008302/1.005858/1.011549 and std0.007319/0.00443663/0.00903018. Step1840 means1.010145/1.006571/1.013693 and std0.00912552/0.00499132/0.0108387. Alpha is active but remains near shared mixing; no strong learned position specialization demonstrated. The whole V2 network was jointly retrained, so its accuracy difference cannot be attributed solely to terminal alpha magnitude.
- No saved c, per-layer b_l or per-position shifts: sample-level dynamic-offset differences cannot be recovered without a new forward, which was not performed. Do not infer a failed gradient path, prove that extra freedom is unnecessary, or conclude that higher alpha LR would help. Initializing multiplicative unit-scale alpha at1 with LR1e-4 may limit finite-budget exploration, but optimization constraint versus task preference is not distinguished here.
- Checkpoint: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_layer_mix_prefix/pathvqa_v2_layer_mix_prefix_p20_norm_fixed_seed44_20260925/checkpoints/epoch_3/visual_selection_layer_mix.pt`; trajectory `v2_diagnostics.jsonl` at run root. Audit output: same run root plus `diagnostics/alpha_readonly_20260925_213130`. Full20x3 matrix in user-provided audit report. Preserve V1 reference and leave V2 closed without automatic LR scans/new seeds.

### 2026-09-26 - V3 postvisual conditional P20 seed44 completed (no demonstrated improvement)

- Exact experiment: `pathvqa_v3_postvisual_prefix_p20_norm_fixed_seed44`; PathVQA model seed44/data seed42, matched3 epochs and fixed epoch3 complete Validation6259 questions/832 images. Controlled change from normalized V1: move same20 conditional Prompt tokens after complete image segment and before real question, retain architecture and1,864,963 trainable parameters. Reported Git commit `e4662ca2fb0872bee13d2babd7de1346c4fd1799`. User-pasted result, no independent raw artifact retrieval. Exact run output directory not supplied; do not infer timestamp/suffix. No new Test/other-seed results supplied.
- Overall58.2361,Yes-No90.5920,Free-form25.9732. Types how7.7519,other14.2857,what21.0754,when0,where64.7922,why0,yes/no90.5920. Standalone clustered95%CI[56.7638,59.6615],iterations2000,seed42,832 image clusters.
- V3 minus normalized V1 paired bootstrap10000,seed42: Overall -0.3355,V3-only255/V1-only276,CI[-1.075118120428111,0.39301247837220044],clusters832; Yes-No +0.3840,116/104,CI[-0.541234564309921,1.343161880396546],clusters810; Free-form -1.0530,139/172,CI[-2.1592232954049875,0.031271749688334106],clusters821; what -0.9027,106/129,CI[-2.0875748119096253,0.2773513217140853],clusters817; where -1.9560,31/39,CI[-5.882352941176471,1.9607843137254901],clusters408. Deltas copied from unrounded calculation as reported. Subgroups exploratory. All intervals include0: no established gain/harm/equivalence, and no proof of noninferiority.
- Training runtime6477.7823sec; peakGPU25010671104bytes; inference timing/final trajectory not supplied. Net21 fewer correct than V1,12 more Yes-No and33 fewer Free-form. V1 remains reference; do not promote V3 or auto-add seeds/position scans. Structural inability of front Prompt to read later vision tokens exists, but removing it did not improve this tested run; not an established performance bottleneck.
- Mechanism boundary: condition-question mismatch still supplies evidence of checkpoint dependence; it does not establish necessity of question conditioning after retraining or locate a remediable defect. A wrong condition can be worse than a model trained without that condition. V3 does not invalidate previous mismatch measurements; they address different questions. Next work should target observed optimization/generalization/answer errors, not more interface changes without evidence.

### 2026-09-26 - Normalized V1 seed44 stratified fitting audit completed

- Exact target: `pathvqa_v1_norm_fixed_seed44_fit_audit`; fixed normalized V1 seed44/data seed42 epoch3 checkpoint, diagnostic commit `b845e07d1e9b3ea7ab6981e4eac23dff7d62695f`, PyTorch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0. Read-only scope: fixed seed42 Train and Validation samples, answer-free greedy generation, teacher-forced CE, existing trajectory parsing and exported error-review examples. No training, parameter update, optimizer step, checkpoint write, Test, or additional seed. Validation predictions were reused. Output: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_norm_fixed_seed44_fit_audit_20260926`; manifest SHA256 `5ba0ae685ff6570e509f37f93a7990f4050a848d96883ca001082fa27dfda930`.
- Sampling is exactly256 questions and256 distinct images per split: Yes/No96, what96, where48, other16, no shortage. These stratified totals are **not official Overall**. Train/Validation sampled accuracy: total61.7188/53.1250; Yes/No96.8750/87.5000; what25.0000/20.8333; where75.0000/64.5833; other31.2500/6.2500 (other n=16 each). Corresponding answer-body question-equal CE: total0.9572/1.2061; Yes/No0.1346/0.2978; what1.8493/2.1466; where0.3484/0.6361; other2.3672/2.7226. Token-equal total CE is1.7514/1.8038. Template CE is small (about0.05 total), so the difficulty is concentrated in answer content rather than end/template tokens.
- Answer-length stratification, Train/Validation: one-token accuracy91.9118/82.2581 with CE0.3060/0.4999; two-token26.0870/35.4839 with CE1.8146/1.9281; 3+ token27.8351/22.7723 with CE1.6670/1.8589. Length is strongly associated with task/type and frequency, so this is descriptive rather than an isolated length effect.
- Training-answer frequency stratification shows a dominant frequency effect. For answers occurring1,2-5,6-20,21+ times in Train, sampled Train accuracy is2.7027/8.3333/33.3333/84.1808 with question-equal CE2.7219/2.6158/1.3820/0.3184; Validation accuracy is0/0/15.3846/74.8603 with CE2.3958/2.3615/2.2056/0.4868. These listed Validation buckets cover217/256; the existing summary omitted an explicit bucket for39 answers unseen in Train, so their aggregate accuracy/CE is **missing**, not inferred. The frequency association is confounded by answer type/length and is not itself causal.
- Split audit found zero overlapping image IDs but37 exact normalized question-answer strings across different images, mostly generic Yes/No, organ/system and `what is present`/`where is this` patterns. This is not image leakage; it does mean wording/label repetition contributes to both splits.
- Existing training trajectory has92 Trainer log records and93 diagnostic records; early checkpoints are absent. Epoch loss means0.88219,0.70997,0.64900; epoch3 begins0.7253 and ends0.6235. Last20% loss mean0.63887 (p10/p50/p90 0.5845/0.6415/0.6902) with preclip total grad-norm mean0.83881. Offset/P20 RMS grows from epoch1 mean0.1518 to epoch2 0.3445 and epoch3 0.3598; last20% mean0.3532. All trainable groups retain finite post-global-clip gradients. Loss continues downward without a late upturn, but no intermediate Validation checkpoints exist, so this alone neither proves more epochs will improve Validation nor excludes emerging overfit.
- Fixed Free-form review set was20 wrong+5 correct from each split. The user stopped further image review after numeric evidence was sufficient. Already inspected/prediction-text evidence contains many substantive organ/system/pathology errors (for example liver→gastrointestinal/endocrine, esophagus→colon, lung→brain) plus several granularity or official exact-match near misses (`due to vertebral column trauma`→`due to trauma`, liver→hepatobiliary, detailed metastatic-spleen description→`spleen`). Complex histology remains medically uncertain. Therefore scoring/supervision wording contributes, but it does not explain the broad low Train what accuracy or the majority of clear content errors.
- Interpretation: this sample does **not** support ordinary severe Train-fit/Validation-generalization overfit as the leading limitation. Open-ended identification is already weak on seen training examples (what25.0%; non-Yes/No groups are not close to saturation), and rare training answers are barely learned. Train→Validation gaps remain present (total8.59 points; what4.17; where10.42; Yes/No9.38), so generalization is also imperfect. Evidence most strongly supports residual optimization/sample-efficiency underfitting, with answer-frequency imbalance and exact-match semantics as secondary limitations.
- Single priority recommendation: before changing structure or supervision, run one controlled **longer normalized V1 seed44 training** experiment (suggested5 epochs from scratch, same architecture/data/objective/LR groups and fixed evaluation protocol, with schedule extended consistently to the new budget). This is the lowest-complexity test of the observed continuing loss decline and poor Train open-answer fit. It may improve memorization more than rare-answer Validation generalization; therefore the result must be judged on full Validation plus the same fixed Train/Validation probe, not training loss alone. This is a recommendation only; no new run is authorized or started.

### 2026-09-26 - V1 seed44 five-epoch controlled training completed

- Exact experiment `pathvqa_v1_norm_fixed_5ep_seed44`; PathVQA model seed44/data seed42, from-scratch budget3->5 epochs with3% warmup and linear decay extended to new endpoint; architecture/learning-rate groups/corrected accumulation otherwise intended unchanged. Primary checkpoint fixed epoch5, not selected best epoch. Trainable1,864,963 verified in pasted audit. Source user attachment; final training commit/runtime/peak memory not included. No Test/other seeds. Output root `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2`, checkpoint `checkpoints/epoch_5`, predictions/summary `eval_validation/epoch_5/pathvqa_predictions.json` and `pathvqa_summary.json`.
- Validation6259/832 images: **59.3386 Overall /90.8800 Yes-No /27.8877 Free-form**. Types how10.8527,other14.2857,what22.0565,when0,where72.3716,why0,yes/no90.8800. Standalone clustered95%CI[57.91,60.75],832clusters; standalone bootstrap settings not supplied.
- Paired5ep minus3ep,10000 cluster resamples,seed42: Overall+0.7669,5-only304/3-only256,CI[0.0,1.537031889946628],n6259/clusters832; Yes-No+0.6720,128/107,CI[-0.25682389168782616,1.607744935944901],n3125/clusters810; Free-form+0.8615,176/149,CI[-0.38684719535783363,2.0867285649537948],n3134/clusters821; what+0.0785,122/120,CI[-1.222891749134843,1.3297341201714699],n2548/clusters817; where+5.6235,49/26,CI[1.4705882352941178,9.75609756097561],n409/clusters408. Overall lower endpoint equals0, not strictly positive. Subgroups exploratory. Net48 correct gain includes21 Yes-No,27 Free-form; where contributes23 net and what2. Does not show broad resolution of open-ended what/long-tail learning.
- Timing seconds TTFT n6259 mean0.0528,p500.052467,p950.054904,min0.037625,max0.350456; TPOT0.020013,49.969tokens/sec; request mean0.106394,p500.093224,p950.155598,min0.063949,max0.427106. Not controlled against earlier runtime/hardware.
- Fit audit completed at `fit_audit_epoch_5`, declared256 train+256 validation, no training/Test. Pasted manifest SHA256 `67a55e35d0dcbcb9b150bf34336c3156f36a1cbccac240fc93c01d32da0ec330` differs from original `5ba0ae685ff6570e509f37f93a7990f4050a848d96883ca001082fa27dfda930`; pending CPU-only identity comparison of split/question/image IDs and reference content to establish identical rows. Serialization/metadata differences can change hash, so do not infer resampling solely from mismatch. Following comparisons are as reported, pending that check; full Validation conclusions unaffected.
- Fit Train groups Yes-No/what/where/other/sample-total accuracy3ep->5ep:96.8750->94.7917,25->27.0833,75->81.25,31.25->25,61.7188->62.5. Body question-equal CE respectively0.134619->0.106542,1.849292->1.556281,0.348401->0.206891,2.367175->2.107551,0.957240->0.794073.
- Fit Validation corresponding accuracy87.5->90.625,20.8333->23.9583,64.5833->70.8333,6.25->6.25,53.125->56.6406. Body CE0.297815->0.282153,2.146645->2.014504,0.636146->0.674273,2.722591->2.396509,1.206112->1.137455. These stratified sample totals are not official Overall; where accuracy improved despite worse mean reference CE. No new frequency-bucket statistics supplied, so no claim rare-answer generalization fixed. PIL truncated-read warning occurred; no reported interruption, audit completed all rows.
- Interpretation: positive point-estimate evidence for longer budget/schedule on seed44 without more parameters; stronger than V2/V3 exploratory results, but not proof of guaranteed further-epoch gains or improved multi-seed mean. Preserve3ep three-seed reference and5ep single-seed candidate separately. Priority candidate is same5ep protocol seed45 to check repeatability rather than immediately extending to8/10 epochs; no GPU operation launched here.
- Correction to the fit-audit caveat above: the new manifest file hash changed because the replay artifact records source-manifest metadata, but the completed `compare_pathvqa_v1_fit_audits.py` comparison requires identical sets of256 question IDs per split and verifies question, reference, image ID and stratum for every row before producing the reported deltas. Therefore the compared sample contents were fixed; the differing whole-file SHA is not evidence of resampling.

### 2026-09-26 - Five-epoch V1 seed45 partial result received

- Associated with currently authorized `pathvqa_v1_norm_fixed_5ep_seed45` by conversation/plan context; pasted JSON omits explicit experiment/seed/output identity, pending full summary confirmation. PathVQA Validation, model seed45/data seed42, intended matched5 epochs/fixed epoch5; same normalized V1 architecture1,864,963 parameters. Controlled variable versus5ep seed44 is initialization; versus3ep seed45 is training budget and corresponding schedule. User reports **58.5077 Overall /90.4320 Yes-No /26.6752 Free-form**. Per-question scores, paired CI, diagnostics, runtime, executing commit and exact output path not provided. Do not invent timestamp/suffix. No GPU operations executed by assistant.
- Conditional on this being planned5ep seed45: versus rounded3ep57.17/88.9920/25.43, changes approximately+1.34/+1.4400/+1.25. Five-epoch seeds44/45 Overall59.3386/58.5077 give mean58.92315 (report58.9232),gap0.8309; Free-form mean27.28145. Three-epoch same two seeds mean approximately57.87,gap1.40. Both observed matched seeds improve, supporting continuation to an unchanged5ep seed46 candidate rather than new structure/LR scan, but this message does not authorize a new GPU run. Two-seed summary does not establish variance reduction or replace full3-seed reporting.

### 2026-09-27 - Five-epoch V1 seed46 result and three-seed summary

- Associated with planned `pathvqa_v1_norm_fixed_5ep_seed46` by current authorized plan/conversation; pasted result omits explicit identity/path, pending full report confirmation. PathVQA model seed46/data seed42, intended matched5 epochs/fixed epoch5, normalized V1 with1,864,963 trainable parameters; only seed differs from five-epoch44/45. User supplies **58.7314 Overall /90.3360 Yes-No /27.2176 Free-form**. Types how10.0775,other7.1429,what21.8603,when0,where68.7042,why4.7619,yes/no90.3360. Standalone95%image-cluster CI[57.2346,60.0702],iterations2000,seed42,832clusters. Exact output path, executing commit, runtime/memory, paired3ep comparison and fit audit not supplied; do not invent these. Assistant performed CPU-only ledger/statistics work, no GPU operations.
- Five-epoch seeds44/45/46 reported59.3386/58.5077/58.7314: Overall **58.8592 +/-0.4299 sample SD**, range0.8309. Free-form27.8877/26.6752/27.2176: **27.2602 +/-0.6074**. Seed45/46 identity association remains based on plan until exact run metadata supplied. No Test scores or variance significance test. All three point estimates exceed respective3ep V1 scores; seed46 gains approximately1.15 Overall and2.14 Free-form versus rounded3ep57.58/25.08.
- Compared with historical final QDPT Sandwich59.1522 +/-1.7528, mean is0.2930 points lower, observed SD lower, parameter count76.11% lower (1.865M versus7.805M), but five versus three epochs and historical normalization/runtime provenance differ. Old all-after placement58.9658 +/-0.4799 also exists; do not frame V1 as uniquely low-variance against only the high-variance ordering. No noninferiority/equivalence claim from this descriptive comparison. Current V1 is1.865M, not1M. Distance to59 mean is0.1408 points; compression to1M remains untested and cannot be promised score-neutral.

### 2026-09-27 - V1 unified Visual20 seed44 first launch failed before training

- Exact experiment target: `pathvqa_v1_visual20_lr1e4_norm_fixed_5ep_seed44`; PathVQA model seed44/data seed42, intended normalized five-epoch Visual20 experiment. The user-launched first attempt failed during model construction before real-batch GPU preflight or any optimizer step, so it has no checkpoint or Validation score. Exact allocated output suffix/path was not included in the returned excerpt and is not inferred.
- Failure: after the subclass consolidated and deleted `visual_s8`/`visual_av10`, its ready-state `trainable_parameter_groups()` called the parent implementation before removing the old group keys. The parent immediately dereferenced the deleted `visual_s8`, raising `AttributeError: 'VisualSelectionPrefixVisual20Model' object has no attribute 'visual_s8'`. Initialization audits before that point showed the expected original V0/V1 seed44 tensors; they are not performance results.
- Fix: the Visual20 subclass now directly returns the exact six intended groups (`p20`, `visual_prompt20`, `question_context`, `maps`, `layer_condition`, `prefix_output`) instead of calling the split-Visual18 parent grouping after conversion. No architecture, initialization, learning rate, schedule, data, evaluation or parameter-budget choice changed. Retry remains user-executed; the failed attempt must not enter score tables.

### 2026-09-27 - Unified Visual20 five-epoch seed44 completed (negative result)

- Exact experiment: `pathvqa_v1_visual20_lr1e4_norm_fixed_5ep_seed44`; PathVQA model seed44/data seed42, fixed epoch5 Validation. Joint change from five-epoch normalized V1: layer-index17 Visual18 S8@3e-5 + Av10@1e-4 replaced by one Visual20@1e-4. Expected/code-audited parameters1,867,011; runtime parameter-audit excerpt not supplied. Local implementation preserves first18 initial rows and shared initial tensors, initializes extra2 using a private CPU generator with global RNG assertions. User result does not include those runtime audit lines. Executing commit/training runtime/peak memory not supplied.
- Output `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual20_lr1e4_norm_fixed_5ep_seed44_20260927_1`; checkpoint `checkpoints/epoch_5`; predictions/summary `eval_validation/epoch_5/pathvqa_predictions.json` and `pathvqa_summary.json`. Training completed following separately recorded construction-failure fix. No Test or other seeds.
- Validation6259 questions/832 images: Overall57.7249, Yes-No90.5920, Free-form24.9521. Types how10.0775,other14.2857,what20.4867,when0,where59.6577,why4.7619,yes/no90.5920. Standalone image-cluster95%CI[56.25,59.06]; standalone bootstrap count/seed not supplied.
- Paired versus five-epoch V1 seed44, image-cluster10000 resamples seed42, variant minus baseline: Overall59.3386->57.7249 delta-1.6137, variant-only230/baseline-only331, CI[-2.427689989764367,-0.8013542889149685], n6259/clusters832. Yes-No90.8800->90.5920 delta-0.2880,107/116,CI[-1.2428397325208105,0.6707283141687651],n3125/clusters810. Free-form27.8877->24.9521 delta-2.9355,123/215,CI[-4.2262279642059495,-1.6702244159201411],n3134/clusters821. what22.0565->20.4867 delta-1.5699,103/143,CI[-2.886164724567322,-0.2742919848429594],n2548/clusters817. where72.3716->59.6577 delta-12.7139,15/67,CI[-16.87041564792176,-8.557457212713937],n409/clusters408. Deltas preserved as reported from unrounded scores. Subgroups exploratory.
- Timing seconds: TTFT n6259 mean0.048788,p500.04895,p950.051408,min0.030001,max0.325883; TPOT0.016146,61.935tokens/sec; request mean0.092351,p500.082151,p950.134934,min0.050667,max0.552406. No controlled hardware timing comparison.
- Interpretation: clear validation degradation for this checkpoint/configuration, dominated by Free-form (92 of101 net lost correct answers; where52 and what40). Does not isolate token-count effect: first8 visual rows also changed LR3e-5->1e-4. Does not establish18 as an intrinsically special count or prove two semantic roles. Preserve original five-epoch V1. Candidate single-variable bridge is18 tokens all@1e-4: versus original isolates LR; versus Visual20 isolates added rows/count at matched LR. Proposed only, not implemented or launched. Deep-per-layer and combined configurations remain untested; do not automatically proceed with combined Visual20.

### 2026-09-27 - Deep Visual5 layers16-23 five-epoch seed44 completed (negative result)

- Exact experiment `pathvqa_v1_deep_visual5_l16_23_norm_fixed_5ep_seed44`; PathVQA model seed44/data seed42, normalized V1 fixed epoch5. Replaces entire layer17 Visual18 with independent static5 prompts per block at zero-based16-23, inserted/removed within each block; uniform visualLR1e-4, Normal(0,0.02). Planned/code-audited trainable1,887,491; runtime parameter/init audit, executing commit, training runtime and peak memory not present in returned excerpt. Joint layout/count/LR change, not a one-factor test of depth. No Test/other seeds.
- Exact output `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_deep_visual5_l16_23_norm_fixed_5ep_seed44_20260927`; checkpoint `checkpoints/epoch_5`, predictions/summary `eval_validation/epoch_5/pathvqa_predictions.json` and `pathvqa_summary.json`.
- Validation6259 questions/832 images: Overall58.2042, Yes-No89.5680, Free-form26.9304. Types how10.0775,other14.2857,what21.1538,when7.6923,where70.4156,why4.7619,yes/no89.5680. Standalone image-cluster95%CI[56.68,59.60]; standalone iterations/seed not supplied.
- Paired variant minus original five-epoch V1 seed44, image-cluster10000 resamples seed42: Overall59.3386->58.2042 delta-1.1344,variant-only260/baseline-only331,CI[-1.9743189540186785,-0.2986012887002986],n6259/clusters832; Yes-No90.8800->89.5680 delta-1.3120,94/135,CI[-2.270516509395064,-0.35368922772772216],n3125/clusters810; Free-form27.8877->26.9304 delta-0.9572,166/196,CI[-2.3603741961892157,0.393191242506929],n3134/clusters821; what22.0565->21.1538 delta-0.9027,134/157,CI[-2.4125764319287883,0.5969020805188813],n2548/clusters817; where72.3716->70.4156 delta-1.9560,27/35,CI[-5.853658536585366,1.7156862745098038],n409/clusters408. Preserve deltas calculated from unrounded values. Subgroups exploratory.
- Timing seconds TTFT count6259 mean0.0495,p500.049638,p950.052236,min0.030321,max0.311212; TPOT0.015716,63.629tokens/sec; request mean0.093199,p500.082844,p950.139003,min0.050408,max0.397563. Hardware-controlled speed comparison unavailable.
- Interpretation: Overall and Yes-No paired95% intervals below zero for this checkpoint comparison; anticipated Yes-No improvement not observed. Net71 fewer correct, including41 Yes-No and30 Free-form. Free-form/what/where intervals include zero, which does not establish equivalence. Deep5 exceeds Visual20 Overall point estimate by0.4793, but no direct paired comparison provided. Both replacements underperform original; cannot infer18 is intrinsically optimal, split groups have semantic roles, or identify LR versus topology as causal. Checkpoint bootstrap addresses evaluation sample uncertainty, not multi-seed training variance. Keep original five-epoch V1 as main candidate; no automatic combined run. A remaining optional explanatory run is original18 with both groupsLR1e-4, keeping initial values fixed; no new run started.

### 2026-09-27 - Original Visual18 uniform low learning rate five-epoch seed44 completed

- Exact experiment `pathvqa_v1_visual18_uniform_lr3e5_norm_fixed_5ep_seed44`; PathVQA model seed44/data seed42, fixed epoch5 normalized V1. Only intended training change: Av10 LR1e-4->3e-5, retaining S8@3e-5, original18 rows, order, initialization and layer17 insertion/removal. Expected unchanged1,864,963 trainable parameters; executing commit, runtime init/parameter audits, training runtime and peak memory absent from result excerpt. No Test/other seeds.
- Output `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual18_uniform_lr3e5_norm_fixed_5ep_seed44_20260927`; checkpoint `checkpoints/epoch_5`, predictions/summary `eval_validation/epoch_5/pathvqa_predictions.json` and `pathvqa_summary.json`.
- Validation6259 questions/832 images: Overall57.9805, Yes-No88.5760, Free-form27.4729. Types how12.4031,other14.2857,what22.0565,when7.6923,where68.2152,why4.7619,yes/no88.5760. Standalone image-cluster95%CI[56.45,59.41], standalone iterations/seed not supplied.
- Paired variant minus original five-epoch V1 seed44, image-cluster10000 resamples seed42: Overall59.3386->57.9805 delta-1.3580,variant-only265/baseline-only350,CI[-2.150894761650306,-0.5694651213770362],n6259/clusters832; Yes-No90.8800->88.5760 delta-2.3040,82/154,CI[-3.243447564889801,-1.347879789915931],n3125/clusters810; Free-form27.8877->27.4729 delta-0.4148,183/196,CI[-1.7555297581218563,0.8795044352938762],n3134/clusters821; what22.0565->22.0565 delta0,143/143,CI[-1.377144821156041,1.402523699264845],n2548/clusters817; where72.3716->68.2152 delta-4.1565,30/47,CI[-8.51581508515815,0.0],n409/clusters408. Deltas as reported from unrounded scores. where upper endpoint equals0, not strictly below0; subgroup analysis exploratory.
- Timing seconds TTFT n6259 mean0.048301,p500.048458,p950.050795,min0.028886,max0.311557; TPOT0.015563,64.255tokens/sec; request mean0.091439,p500.080972,p950.140486,min0.048637,max0.545145. No controlled hardware speed conclusion.
- Interpretation: matched-layout low-LR control underperforms original for this seed/checkpoint; Overall and Yes-No paired95%CI below0. Of85 net lost correct,72 are Yes-No and13 Free-form. what total unchanged but286 questions switch correctness, so equal accuracy does not mean unchanged behavior. Supports retaining original LR allocation over tested uniform3e-5 at five epochs; does not establish split semantic roles, necessity versus every uniform LR, or that18 is optimal. Lower LR and fixed training budget remain linked: current outputs cannot distinguish slower optimization from beneficial heterogeneous dynamics. Existing uniform-high control uses20 rows and remains count-confounded. Multi-seed robustness of ablation not tested. Deep10+10 stays an untested candidate, not guaranteed rescued by this result. No GPU operation launched by assistant.

### 2026-09-28 - Final exploration: deep per-layer10+10 completed; exploration closed

- Exact experiment `pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44`; PathVQA model seed44/data seed42, normalized V1 from-scratch five epochs, primary fixed epoch5. Replaces original layer17 Visual18 with independent10 low-LR3e-5 +10 high-LR1e-4 prompts at each zero-based block16-23, inserted/removed within each block. Expected/code-audited2,010,371 trainable parameters; runtime count/init audit, executing commit, training runtime and peak memory not supplied in returned evaluation excerpt. No Test or additional seeds.
- User-supplied complete result preserved below, including exact output/checkpoint/predictions/summary paths, all subgroup statistics, both paired comparisons and timing. Standalone CI resampling settings absent; paired comparisons use10000 image-cluster resamples,seed42.
- Overall57.2935 /Yes-No89.4720 /Free-form25.2074. Against original five-epoch V1: Overall-2.0451 CI[-2.891839377995702,-1.2313710989207682], net128 fewer correct (44 Yes-No,84 Free-form). Against Deep5: Overall-0.9107 CI[-1.6792105836686682,-0.14593687456831492], Free-form-1.7230 CI[-2.9940591853190495,-0.45042514424536934], where-9.5355 CI[-14.425427872860636,-4.634146341463414]. Mixed learning rates did not rescue this tested deep configuration; comparison also changes per-layer token count and initialization, so cannot isolate LR or capacity effects. Original8+10 remains empirically preferred among tested settings; no proof that18 or split8/10 is intrinsically optimal, or that the two groups have distinct semantic roles. Uniform-low original18 is the direct matched-layout LR control; other variants change multiple factors. Paired CIs concern fixed checkpoints/evaluation sampling, not multi-seed superiority of all variants.
- As explicitly agreed, this is the final exploration. No further training, retries, tuning, seeds, Test or compression runs scheduled. Retain normalized five-epoch original V1 with1,864,963 parameters and previously reported three-seed mean58.8592/sampleSD0.4299 (seed45/46 full provenance caveats remain in prior records). Ledger updates only, no ledger-only commit/push. Assistant ran CPU-only file operations.

<details>
<summary>Complete user-supplied final experiment output</summary>

```text
========== PathVQA Evaluation ==========
Overall Accuracy: 57.29
Yes/No Accuracy: 89.47
Free-form Accuracy: 25.21
Per Question Type: {'how': 7.7519, 'other': 7.1429, 'what': 20.7221, 'when': 7.6923, 'where': 60.8802, 'why': 4.7619, 'yes/no': 89.472}
Image-clustered 95% CI: [55.81, 58.68] clusters=832
TTFT: {'count': 6259, 'mean': 0.05122, 'p50': 0.05124, 'p95': 0.056029, 'min': 0.03131, 'max': 0.286374}
TPOT: 0.016314 s/token, 61.296 token/s
Request Latency: {'count': 6259, 'mean': 0.097981, 'p50': 0.086085, 'p95': 0.148184, 'min': 0.052093, 'max': 0.333849}
Predictions: /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44_20260928/eval_validation/epoch_5/pathvqa_predictions.json
Summary: /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44_20260928/eval_validation/epoch_5/pathvqa_summary.json
[PATHVQA_PAIRED_COMPARISON] group=overall n=6259 clusters=832 baseline=59.3386 variant=57.2935 delta=-2.0451 variant_only=253 baseline_only=381 ci={'confidence_level': 0.95, 'lower': -2.891839377995702, 'upper': -1.2313710989207682, 'iterations': 10000, 'seed': 42, 'image_clusters': 832}
[PATHVQA_PAIRED_COMPARISON] group=yes_no n=3125 clusters=810 baseline=90.8800 variant=89.4720 delta=-1.4080 variant_only=84 baseline_only=128 ci={'confidence_level': 0.95, 'lower': -2.2955182696627925, 'upper': -0.5403558521792307, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
[PATHVQA_PAIRED_COMPARISON] group=free_form n=3134 clusters=821 baseline=27.8877 variant=25.2074 delta=-2.6803 variant_only=169 baseline_only=253 ci={'confidence_level': 0.95, 'lower': -4.096656956753147, 'upper': -1.2867313825662108, 'iterations': 10000, 'seed': 42, 'image_clusters': 821}
[PATHVQA_PAIRED_COMPARISON] group=what n=2548 clusters=817 baseline=22.0565 variant=20.7221 delta=-1.3344 variant_only=138 baseline_only=172 ci={'confidence_level': 0.95, 'lower': -2.8275587415372363, 'upper': 0.19107127432335272, 'iterations': 10000, 'seed': 42, 'image_clusters': 817}
[PATHVQA_PAIRED_COMPARISON] group=where n=409 clusters=408 baseline=72.3716 variant=60.8802 delta=-11.4914 variant_only=27 baseline_only=74 ci={'confidence_level': 0.95, 'lower': -16.176470588235293, 'upper': -6.862745098039215, 'iterations': 10000, 'seed': 42, 'image_clusters': 408}
[PATHVQA_PAIRED_COMPARISON] group=question_type:how n=129 clusters=104 baseline=10.8527 variant=7.7519 delta=-3.1008 variant_only=2 baseline_only=6 ci={'confidence_level': 0.95, 'lower': -9.21985815602837, 'upper': 1.6666666666666667, 'iterations': 10000, 'seed': 42, 'image_clusters': 104}
[PATHVQA_PAIRED_COMPARISON] group=question_type:other n=14 clusters=8 baseline=14.2857 variant=7.1429 delta=-7.1429 variant_only=0 baseline_only=1 ci={'confidence_level': 0.95, 'lower': -23.076923076923077, 'upper': 0.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 8}
[PATHVQA_PAIRED_COMPARISON] group=question_type:when n=13 clusters=12 baseline=0.0000 variant=7.6923 delta=+7.6923 variant_only=1 baseline_only=0 ci={'confidence_level': 0.95, 'lower': 0.0, 'upper': 25.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 12}
[PATHVQA_PAIRED_COMPARISON] group=question_type:why n=21 clusters=20 baseline=0.0000 variant=4.7619 delta=+4.7619 variant_only=1 baseline_only=0 ci={'confidence_level': 0.95, 'lower': 0.0, 'upper': 15.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 20}
[PATHVQA_PAIRED_COMPARISON] group=question_type:yes/no n=3125 clusters=810 baseline=90.8800 variant=89.4720 delta=-1.4080 variant_only=84 baseline_only=128 ci={'confidence_level': 0.95, 'lower': -2.2955182696627925, 'upper': -0.5403558521792307, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
[PATHVQA_PAIRED_COMPARISON] group=overall n=6259 clusters=832 baseline=58.2042 variant=57.2935 delta=-0.9107 variant_only=290 baseline_only=347 ci={'confidence_level': 0.95, 'lower': -1.6792105836686682, 'upper': -0.14593687456831492, 'iterations': 10000, 'seed': 42, 'image_clusters': 832}
[PATHVQA_PAIRED_COMPARISON] group=yes_no n=3125 clusters=810 baseline=89.5680 variant=89.4720 delta=-0.0960 variant_only=111 baseline_only=114 ci={'confidence_level': 0.95, 'lower': -0.9964801849112586, 'upper': 0.7840639032483296, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
[PATHVQA_PAIRED_COMPARISON] group=free_form n=3134 clusters=821 baseline=26.9304 variant=25.2074 delta=-1.7230 variant_only=179 baseline_only=233 ci={'confidence_level': 0.95, 'lower': -2.9940591853190495, 'upper': -0.45042514424536934, 'iterations': 10000, 'seed': 42, 'image_clusters': 821}
[PATHVQA_PAIRED_COMPARISON] group=what n=2548 clusters=817 baseline=21.1538 variant=20.7221 delta=-0.4317 variant_only=143 baseline_only=154 ci={'confidence_level': 0.95, 'lower': -1.7932258097398872, 'upper': 0.9286119141268462, 'iterations': 10000, 'seed': 42, 'image_clusters': 817}
[PATHVQA_PAIRED_COMPARISON] group=where n=409 clusters=408 baseline=70.4156 variant=60.8802 delta=-9.5355 variant_only=36 baseline_only=75 ci={'confidence_level': 0.95, 'lower': -14.425427872860636, 'upper': -4.634146341463414, 'iterations': 10000, 'seed': 42, 'image_clusters': 408}
[PATHVQA_PAIRED_COMPARISON] group=question_type:how n=129 clusters=104 baseline=10.0775 variant=7.7519 delta=-2.3256 variant_only=0 baseline_only=3 ci={'confidence_level': 0.95, 'lower': -5.147058823529412, 'upper': 0.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 104}
[PATHVQA_PAIRED_COMPARISON] group=question_type:other n=14 clusters=8 baseline=14.2857 variant=7.1429 delta=-7.1429 variant_only=0 baseline_only=1 ci={'confidence_level': 0.95, 'lower': -23.076923076923077, 'upper': 0.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 8}
[PATHVQA_PAIRED_COMPARISON] group=question_type:when n=13 clusters=12 baseline=7.6923 variant=7.6923 delta=+0.0000 variant_only=0 baseline_only=0 ci={'confidence_level': 0.95, 'lower': 0.0, 'upper': 0.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 12}
[PATHVQA_PAIRED_COMPARISON] group=question_type:why n=21 clusters=20 baseline=4.7619 variant=4.7619 delta=+0.0000 variant_only=0 baseline_only=0 ci={'confidence_level': 0.95, 'lower': 0.0, 'upper': 0.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 20}
[PATHVQA_PAIRED_COMPARISON] group=question_type:yes/no n=3125 clusters=810 baseline=89.5680 variant=89.4720 delta=-0.0960 variant_only=111 baseline_only=114 ci={'confidence_level': 0.95, 'lower': -0.9964801849112586, 'upper': 0.7840639032483296, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
experiment      seed    protocol        validation_epoch        validation_accuracy     checkpoint
pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44  44      fixed_epoch5_validation 5       57.2935 /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44_20260928/checkpoints/epoch_5
[PATHVQA_V1_DEEP20_SPLIT_LR_DONE] output=/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44_20260928 primary_epoch=5 baselines=v1_59.3386,deep5_58.2042 final_exploration=true test_evaluation=false other_seeds=false
[DONE] 已完成实验目标: pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44
```
</details>

### 2026-10-09：PathVQA V10三seed五轮Validation完成（负结果；自动关机失败）

实验名：`pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44`、`pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed45`、`pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed46`。数据集PathVQA，固定epoch5完整Validation，6259题/832图；三项按44→45→46执行。受控改动为保留问题编码、三层地图及门控、Visual18/P20，将三份独立LN/64维投影及192维拼接输出头，改为先融合地图、读取单份共同merger Value，再以2560→160→2560 Meta-Net生成共享偏移。沿用已授权V1五轮协议（data seed42、batch2/accum16、归一化修正）；本条成绩及统计来自用户回传，不代表已独立核验远程训练元数据。

| seed | Overall（完整精度） | Yes/No | Free-form | how | other | what | when | where | why |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 44 | 57.70889918517335 | 89.632 | 25.8775 | 9.3023 | 7.1429 | 20.8006 | 0 | 65.2812 | 4.7619 |
| 45 | 55.82361399584598 | 87.904 | 23.8354 | 8.5271 | 14.2857 | 19.3485 | 0 | 58.6797 | 4.7619 |
| 46 | 56.73430260424988 | 89.696 | 23.8673 | 10.8527 | 7.1429 | 19.1915 | 0 | 59.4132 | 4.7619 |

同seed原V1 epoch5配对（Overall主指标，非探索性；10000次图像簇bootstrap、seed42、95%置信度；各6259题/832簇）：

| seed | V1 Overall | V10−V1（pp） | V10独占/V1独占 | McNemar exact p | 配对CI（完整精度） |
| --- | ---: | ---: | ---: | ---: | --- |
| 44 | 59.33855248442243 | -1.6296532992490782 | 285/387 | 9.498132042054199e-05 | [-2.492766371838962, -0.7465026924460755] |
| 45 | 58.507748841668 | -2.6841348458220153 | 267/435 | 2.4107166301275105e-10 | [-3.594676850903095, -1.775974131810465] |
| 46 | 58.7314267454865 | -1.9971241412366183 | 265/390 | 1.1791008224579698e-06 | [-2.854083815283294, -1.1415982749723494] |

三seed均值±样本标准差(ddof=1)：Overall56.7556±0.9428，Yes/No89.0773±1.0166，Free-form24.5267±1.1699；how9.5607±1.1841，other9.5238±4.1239，what19.7802±0.8872，when0±0，where61.1247±3.6183，why4.7619±0，yes/no89.0773±1.0166。原V1 Overall58.8592±0.4299、Free-form27.2602±0.6074；平均Overall下降2.1036pp。三seed配对CI均低于零，不能归为seed44偶然退化；这三个seed的分散程度也没有改善。

修正解释：SLAKE seed44上的简化收益不能推广为原V1头部普遍过度设计。先融合再映射同时改变层身份保留、归一化、瓶颈与优化路径，本次不能独立归因于某一个因素，也不证明两个分别训练的数据集“不相容”。保持原V1为PathVQA参考，不自动追加训练。实际带时间戳输出路径、checkpoint身份、训练耗时/显存及激活诊断本次未回传，待补；预定产物根为`pathvqa/outputs/v10/suite_<timestamp>/`，不得据此猜测实际目录。

自动关机记录：`status=call_failed`，`delay_seconds=600`，`scheduled_unix=1791501232.8998308`，`worker_pid=31559`，`called_at=2026-10-09T07:13:52.899933`，`error=[Errno 8] Exec format error: '/usr/bin/shutdown'`。不能记为已关机。助手未执行GPU或远程关机操作；本条账本保持本地，等待下一次相关代码提交。

### 2026-09-29 - V1A naming correction and V1B implementation prepared

- Explicit naming correction: completed experiment `pathvqa_v1_direct_summary_norm_fixed_5ep_seed44` (879,364 trainable parameters) is now called **V1A**. This adds a name only; its recorded architecture, score, diagnostics and output path are unchanged.
- Prepared, not yet run: **V1B** experiment `pathvqa_v1b_evidence_token_norm_fixed_5ep_seed44`, PathVQA model seed44/data seed42, five epochs with fixed epoch5 Validation. Controlled change versus the normalized five-epoch V1 is only the injection interface: the original full condition head still produces `b=W ReLU(c)+d`, but the sequence becomes `[P20; b; native chat]`; P20 is no longer shifted. Trainable parameter budget remains 1,864,963. There is no score or output path yet.
- V1B preflight is specified to audit 21 prefix positions, attention/labels, mRoPE, visual masks, DeepStack alignment, native embedding preservation, finite evidence-token RMS, all branch gradients, single prefill injection, KV-cache reuse and checkpoint roundtrip. Normal and explicit auto-shutdown launch targets are separate. The shutdown target records stage status and final exit code before scheduling shutdown and records a failed shutdown call.
- Planned comparisons: V1B versus original five-epoch V1 seed44 (59.3386) with image-cluster paired bootstrap 10,000/seed42, and V1B versus V1A when both artifacts are present. V1A versus V1B is not a single-factor comparison because both mapping and injection interface differ. No Test or other seed is scheduled.

### 2026-09-28 - SLAKE normalized V1 five-epoch three-seed results received

- Associated with implemented plan target `slake_v1_norm_fixed_5ep_seeds44_45_46_test`: frozen normalized V1,1,864,963 trainable parameters, model seeds44/45/46 and data seed42, official all-language SLAKE Test, fixed epoch5, no Validation/checkpoint selection. Association is based on plan plus user context; pasted scores do not include exact per-run names, timestamped output paths, sample counts, executing commit, runtime or audits. These provenance fields remain pending and must not be invented. Controlled change versus final PathVQA V1 is dataset/evaluator migration; old QDPT comparisons also differ in architecture and3vs5-epoch protocol.

| Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|44|75.5500|82.6600|70.8300|55.4300|78.4900|77.2900|73.7700|
|45|76.3100|82.6600|72.1000|56.5500|79.2000|77.7600|74.8300|
|46|78.3700|85.7700|73.4500|60.3000|81.0100|78.0400|78.7000|
|Mean|76.7433|83.6967|72.1267|57.4267|79.5667|77.6967|75.7667|
|Sample SD|1.4591|1.7956|1.3102|2.5506|1.2994|0.3790|2.5950|

- Means/sampleSD recomputed from supplied rounded scores and agree with supplied summary. Overall range2.82; EN range0.75,ZH4.93,KVQA4.87. These are overlapping partitions, so high ZH/KVQA variability does not independently identify a causal instability source. No paired predictions or CIs supplied. Do not select only seed46 or confuse its78.37 with historical Full Workspace seed44 same rounded score.
- Historical final Sandwich QDPT SLAKE has76.74/76.65/77.46,mean76.95/sampleSD0.44396,7,805,184 parameters. New minus old mean-0.2067 points; parameter reduction76.1061%. Descriptively similar mean with substantially fewer parameters, but no equivalence/noninferiority claim and observed seedSD is higher, not improved. Historical3ep normalization/provenance differences preclude a clean one-factor comparison. KVQA mean62.17->57.4267 (-4.7433), VQA79.11->79.5667 (+0.4567), subject to old rounded means/protocol caveats. Historical LoRA81.82 remains higher than new mean by5.0767, not a new matched-protocol paired comparison.
- Conclusion: cross-dataset utility/parameter-efficiency evidence, not cross-dataset stability improvement or universal LoRA parity. PathVQA reported58.8592+/-0.4299 and SLAKE76.7433+/-1.4591 must both be reported. Frozen configuration remains fixed; no new structure or hyperparameter search prompted by Test results. Next work is provenance and already planned baseline/electrical/ablation evidence, not extra SLAKE tuning. Assistant performed CPU-only reading/statistics/ledger edits.

### 2026-09-29 - RSVQA-LR untrained base Qwen3-VL Test

- Experiment: `rsvqa_lr_base_qwen3vl_test`; dataset: official-layout RSVQA-LR Test; model seed: not applicable because no trainable/randomly initialized method parameters; training: none; backend: unmodified base Qwen3-VL; implementation commit: `124d3bf4e6f178ff3ab0181c983c3e500bc53481`.
- Protocol: all 10,004 active Test questions over 100 image clusters; each images/questions/answers JSON independently filtered by `active=true` and strictly joined by IDs. Generative answers use short-answer prompts. LR count answers use the original `VocabEncoder(range_numbers=True)` categories before exact matching. OA is question-weighted; AA is the unweighted macro mean of the four question-type accuracies.
- Result: **OA 57.5770**, **AA 58.8947**. Per type: rural/urban 69.0000, presence 63.0457, count 29.8948, comparison 73.6382. Image-clustered OA 95% CI [56.3944,58.7048],100 clusters.
- Output: `/root/autodl-tmp/Qwen3-VL-modify-test/RSVQA/outputs/rsvqa_lr_base_qwen3vl_test_20260929`; predictions `rsvqa_predictions.json`, summary `rsvqa_summary.json`. No checkpoint, training, Test selection, extra seed, or assistant-started GPU operation was involved. Runtime/timing breakdown was not included in the returned excerpt.
- Interpretation: OA and AA differ because the four question types are imbalanced. Count has the lowest accuracy; its sample frequency affects OA but its weight in AA is exactly one quarter. This is the fixed zero-training reference for later matched RSVQA methods, not evidence about adaptation stability.

### 2026-09-29 - RSVQA-LR frozen V1 throughput configuration seed44 completed

- Exact experiment `rsvqa_v1_norm_fixed_5ep_b4a8_seed44`; RSVQA-LR official-layout train57,223 active questions, fixed epoch5 Test10,004 questions/100 images; model seed44/data seed42. Frozen V1 structure1,864,963 planned/audited parameters, microbatch4/accumulation8 versus original2/16 with same effective32; other intended five-epoch protocol unchanged. Executing commit/runtime/memory/init audit not supplied. No Validation evaluation or other seeds.
- Test OA85.0360, AA86.0812; rural_urban92.0000,presence91.0998,count69.1211,comp92.1039. Image-cluster95%OA CI[83.8016,86.2155],100clusters; bootstrap iterations/seed absent from returned excerpt. Count scoring uses official interval mapping, not exact object-count accuracy.
- Compared with frozen base OA57.5770/AA58.8947, descriptive gains +27.4590 OA/+27.1865 AA; type gains +23.0000/+28.0541/+39.2263/+18.4657 respectively. No paired deltas supplied. Large adaptation gain is not attribution to question-guided maps versus static/CoCoOp, nor evidence of superiority over LoRA; same-protocol trained baselines pending.
- Output `/root/autodl-tmp/Qwen3-VL-modify-test/RSVQA/outputs/visual_selection_prefix/rsvqa_v1_norm_fixed_5ep_b4a8_seed44_20260929`; checkpoint intended`checkpoints/epoch_5`; supplied predictions/summary`eval_test/epoch_5/rsvqa_predictions.json` and`rsvqa_summary.json`. Keep b4a8 identity, do not relabel as b2a16.

### 2026-09-29 - PathVQA ablation A: direct summary shared offset completed

- Exact experiment `pathvqa_v1_direct_summary_norm_fixed_5ep_seed44`; model seed44/data seed42, five epochs/fixed epoch5 Validation. Replaces three2560->64 projections and192->2560 head/ReLU with alpha-weighted sum of per-layer normalized summaries, keeping P20 shared-offset interface, layer maps/gates and Visual18. Expected879,364 trainable parameters versus1,864,963 (-52.8482%); actual parameter audit, initial/final alpha and calibration/gradient trajectories not in supplied attachment. Executing commit/train runtime/peak memory also absent. No Test/other seeds.
- Overall55.9035 /Yes-No89.1200 /Free-form22.7824. Against five-epoch originalV1: Overall-3.4351 CI[-4.2897443186618425,-2.5828465737491957]; Free-form-5.1053 CI[-6.581954810677049,-3.6907327575009874]; Yes-No-1.7600 CI[-2.7104807981805163,-0.8038361777048676]. Net215 fewer correct,160 Free-form and55 Yes-No. Full exact scores, subgroup paired CIs, paths, evaluation settings and timing preserved verbatim below.
- Interpretation: the tested direct normalized-summary/scalar replacement does not preserve original accuracy at half the parameters; supports learned read/write mapping over this specific alternative under the five-epoch seed44 shared-offset protocol. Does not prove that64-dimensional per-layer bottlenecks, ReLU or all985,600 projection parameters are individually necessary/minimal; transformation/capacity/optimization change jointly. Calibration and alpha dynamics remain unverified from this excerpt. Ablation B independent-token result not yet supplied; do not infer it from A or assume it completed. No additional experiment launched.

<details>
<summary>Complete user-supplied ablation A output</summary>

```text
========== PathVQA Evaluation ==========
Overall Accuracy: 55.90
Yes/No Accuracy: 89.12
Free-form Accuracy: 22.78
Per Question Type: {'how': 11.6279, 'other': 0.0, 'what': 18.0926, 'when': 0.0, 'where': 57.9462, 'why': 4.7619, 'yes/no': 89.12}
Image-clustered 95% CI: [54.41, 57.27] clusters=832
TTFT: {'count': 6259, 'mean': 0.050185, 'p50': 0.048934, 'p95': 0.056477, 'min': 0.036452, 'max': 0.423827}
TPOT: 0.020707 s/token, 48.292 token/s
Request Latency: {'count': 6259, 'mean': 0.105241, 'p50': 0.089042, 'p95': 0.175882, 'min': 0.061358, 'max': 0.690515}
Predictions: /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix_direct/pathvqa_v1_direct_summary_norm_fixed_5ep_seed44_20260929/eval_validation/epoch_5/pathvqa_predictions.json
Summary: /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix_direct/pathvqa_v1_direct_summary_norm_fixed_5ep_seed44_20260929/eval_validation/epoch_5/pathvqa_summary.json
[PATHVQA_PAIRED_COMPARISON] group=overall n=6259 clusters=832 baseline=59.3386 variant=55.9035 delta=-3.4351 variant_only=245 baseline_only=460 ci={'confidence_level': 0.95, 'lower': -4.2897443186618425, 'upper': -2.5828465737491957, 'iterations': 10000, 'seed': 42, 'image_clusters': 832}
[PATHVQA_PAIRED_COMPARISON] group=yes_no n=3125 clusters=810 baseline=90.8800 variant=89.1200 delta=-1.7600 variant_only=98 baseline_only=153 ci={'confidence_level': 0.95, 'lower': -2.7104807981805163, 'upper': -0.8038361777048676, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
[PATHVQA_PAIRED_COMPARISON] group=free_form n=3134 clusters=821 baseline=27.8877 variant=22.7824 delta=-5.1053 variant_only=147 baseline_only=307 ci={'confidence_level': 0.95, 'lower': -6.581954810677049, 'upper': -3.6907327575009874, 'iterations': 10000, 'seed': 42, 'image_clusters': 821}
[PATHVQA_PAIRED_COMPARISON] group=what n=2548 clusters=817 baseline=22.0565 variant=18.0926 delta=-3.9639 variant_only=119 baseline_only=220 ci={'confidence_level': 0.95, 'lower': -5.47518025902624, 'upper': -2.4549290372075183, 'iterations': 10000, 'seed': 42, 'image_clusters': 817}
[PATHVQA_PAIRED_COMPARISON] group=where n=409 clusters=408 baseline=72.3716 variant=57.9462 delta=-14.4254 variant_only=22 baseline_only=81 ci={'confidence_level': 0.95, 'lower': -19.024390243902438, 'upper': -9.803921568627452, 'iterations': 10000, 'seed': 42, 'image_clusters': 408}
[PATHVQA_PAIRED_COMPARISON] group=question_type:how n=129 clusters=104 baseline=10.8527 variant=11.6279 delta=+0.7752 variant_only=5 baseline_only=4 ci={'confidence_level': 0.95, 'lower': -4.065040650406504, 'upper': 6.015037593984962, 'iterations': 10000, 'seed': 42, 'image_clusters': 104}
[PATHVQA_PAIRED_COMPARISON] group=question_type:other n=14 clusters=8 baseline=14.2857 variant=0.0000 delta=-14.2857 variant_only=0 baseline_only=2 ci={'confidence_level': 0.95, 'lower': -33.333333333333336, 'upper': 0.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 8}
[PATHVQA_PAIRED_COMPARISON] group=question_type:when n=13 clusters=12 baseline=0.0000 variant=0.0000 delta=+0.0000 variant_only=0 baseline_only=0 ci={'confidence_level': 0.95, 'lower': 0.0, 'upper': 0.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 12}
[PATHVQA_PAIRED_COMPARISON] group=question_type:why n=21 clusters=20 baseline=0.0000 variant=4.7619 delta=+4.7619 variant_only=1 baseline_only=0 ci={'confidence_level': 0.95, 'lower': 0.0, 'upper': 15.0, 'iterations': 10000, 'seed': 42, 'image_clusters': 20}
[PATHVQA_PAIRED_COMPARISON] group=question_type:yes/no n=3125 clusters=810 baseline=90.8800 variant=89.1200 delta=-1.7600 variant_only=98 baseline_only=153 ci={'confidence_level': 0.95, 'lower': -2.7104807981805163, 'upper': -0.8038361777048676, 'iterations': 10000, 'seed': 42, 'image_clusters': 810}
{
  "count": 6259,
  "overall_accuracy": 55.9035,
  "yes_no_accuracy": 89.12,
  "free_form_accuracy": 22.7824,
  "per_answer_type_accuracy": {
    "free-form": 22.7824,
    "yes/no": 89.12
  },
  "per_question_type_accuracy": {
    "how": 11.6279,
    "other": 0.0,
    "what": 18.0926,
    "when": 0.0,
    "where": 57.9462,
    "why": 4.7619,
    "yes/no": 89.12
  },
  "free_form_question_type_macro_accuracy": 15.4048,
  "clustered_bootstrap": {
    "point_estimate": 55.9035,
    "confidence_level": 0.95,
    "lower": 54.41,
    "upper": 57.2657,
    "iterations": 2000,
    "seed": 42,
    "image_clusters": 832
  },
  "backend": "visual-selection-prefix",
  "base_model": "/root/autodl-tmp/model",
  "checkpoint": "/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix_direct/pathvqa_v1_direct_summary_norm_fixed_5ep_seed44_20260929/checkpoints/epoch_5",
  "v0_intervention": "normal",
  "v0_question_mask_policy": "independent_raw_question_ids_no_prefill_write_mapping",
  "question_source_policy": "independent_raw_question_ids_no_prefill_write_mapping",
  "dynamic_prompt_component_checkpoint": null,
  "dynamic_prompt_components": [],
  "data_root": "/root/autodl-tmp/dataset/pathVQA",
  "split": "validation",
  "max_new_tokens": 32,
  "temperature": 0.0,
  "answer_mode": "raw",
  "instruction": "short-answer",
  "partial_evaluation": false,
  "timing": {
    "methodology": {
      "warmup_runs_excluded": 3,
      "ttft": "generate start to first generated-token logits ready",
      "tpot": "token-count-weighted interval between later token logits",
      "request": "model interface call including preprocessing and decoding",
      "timing_methods": {
        "cuda-events-logits-ready-v2": 6259
      }
    },
    "successful_requests": 6259,
    "model_timed_requests": 6259,
    "generated_tokens": 18356,
    "subsequent_tokens": 12097,
    "ttft_seconds": {
      "count": 6259,
      "mean": 0.050185,
      "p50": 0.048934,
      "p95": 0.056477,
      "min": 0.036452,
      "max": 0.423827
    },
    "tpot_per_request_seconds": {
      "count": 6259,
      "mean": 0.020747,
      "p50": 0.020327,
      "p95": 0.021885,
      "min": 0.019391,
      "max": 0.094406
    },
    "tpot_weighted_seconds": 0.020707,
    "decode_tokens_per_second": 48.292,
    "generation_seconds": {
      "count": 6259,
      "mean": 0.090412,
      "p50": 0.070994,
      "p95": 0.159819,
      "min": 0.056041,
      "max": 0.671988
    },
    "request_seconds": {
      "count": 6259,
      "mean": 0.105241,
      "p50": 0.089042,
      "p95": 0.175882,
      "min": 0.061358,
      "max": 0.690515
    },
    "model_generated_tokens_per_second": 32.438,
    "end_to_end_generated_tokens_per_second": 27.867
  }
}[PATHVQA_V1_DIRECT_DONE] output=/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix_direct/pathvqa_v1_direct_summary_norm_fixed_5ep_seed44_20260929 primary_epoch=5 baseline=59.3386 test_evaluation=false other_seeds=false
[DONE] 已完成实验目标: pathvqa_v1_direct_summary_norm_fixed_5ep_seed44
```
</details>

### 2026-09-30 - V1B independent evidence token and RSVQA-LR LoRA results returned

- PathVQA `pathvqa_v1b_evidence_token_norm_fixed_5ep_seed44`, model seed44/data seed42, fixed epoch5 Validation. Controlled change: keep full V1 condition head and 1,864,963 planned parameters; replace shared P20 offset with `[P20; b; native chat]`. Returned Overall57.4533, Yes/No90.0160, Free-form24.9840; question types how10.0775, other14.2857, what20.4082, when7.6923, where60.1467, why4.7619, yes/no90.0160. Image-cluster95%CI[55.9935,58.8549], 2000 iterations, bootstrap seed42,832 images. Output `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix_evidence/pathvqa_v1b_evidence_token_norm_fixed_5ep_seed44_20260929`. Versus original five-epoch V1: Overall-1.8853, Yes/No-0.8640, Free-form-2.9037, where-12.2249; paired CI not returned. Against V1A Overall+1.5498, but this changes both mapping and injection and is not a single-factor attribution. Supports retaining original interface for this configuration; does not establish all independent-token designs inferior. Runtime, actual parameter/init audit, commit and activation diagnostics not returned.
- RSVQA-LR `rsvqa_lr_lora_full_model_attention_r8_b4a8_seed44`, seed44. Current local launcher defines data seed42, three epochs/fixed epoch3 Test, batch4/accumulation8, rank8/alpha16/dropout0.05, LR1e-4, 3% warmup/linear, normalized gradient accumulation, all24 visual attention blocks and36 language attention blocks, expected7,077,888 trainable parameters. These are locally checked configuration, not a returned server train_report. Test Overall87.0552, Average87.7162; rural_urban92.0000,presence91.6074,count74.2789,comp92.9785. Image-cluster95%CI[86.0184,88.0812],10000 iterations,bootstrap seed42,100clusters. Output `/root/autodl-tmp/Qwen3-VL-modify-test/RSVQA/outputs/lora/rsvqa_lr_lora_full_model_attention_r8_b4a8_seed44_20260929`. Versus V1 OA+2.0192, AA+1.6350; type differences0/+0.5076/+5.1578/+0.8746. Count uses interval accuracy. V1 expected1,864,963 parameters is26.3491% of LoRA (73.6509% fewer). V1 five epochs versus LoRA three: not equal training exposure; no paired CI, walltime or memory comparison supplied. Single seed does not support a stability comparison. No claim of V1 accuracy parity or superiority.
- Status correction to prior pending entries: V1B and LoRA b4a8 now have returned results; do not rerun. No GPU operations performed here; no new experiment scheduled.

### 2026-09-30 - PathVQA C-static / C-qmap / C-noVisual completed

All three: model seed44/data seed42, normalized five-epoch training from scratch, fixed epoch5 Validation, original V1 seed44 comparator Overall59.3386/Yes-No90.8800/Free-form27.8877. User-returned paired image-cluster intervals below; planned bootstrap10000/seed42, runtime bootstrap metadata/commit and initialization audits not included in excerpt. No Test or additional seeds. Differences are reported values (rounding may differ from subtraction of displayed scores).

| Experiment suffix (prefix pathvqa_v1_, suffix _norm_fixed_5ep_seed44) | Controlled change | Overall | Yes/No | Free-form | Parameters | Training seconds | Peak GPU GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| c_static | Remove full dynamic branch; keep original P20+Visual18 |56.7343|88.9600|24.6011|69632|10531.2984|23.136|
| c_qmap | Replace question-generated map queries with learned sample-independent queries; keep question-conditioned layer gates and all other branches |58.7314|90.8480|26.7071|1815811|10843.682|23.287|
| c_no_visual | Remove Visual18 only; retain P20 and full dynamic branch |57.9805|90.7200|25.3350|1846531|10443.9421|22.108|

Exact output roots:
- `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/ablations/pathvqa_v1_c_static_norm_fixed_5ep_seed44_20260930`
- `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/ablations/pathvqa_v1_c_qmap_norm_fixed_5ep_seed44_20260930`
- `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/ablations/pathvqa_v1_c_no_visual_norm_fixed_5ep_seed44_20260930`

Question types in order how/other/what/when/where/why/yes-no:
- c_static: 6.9767 / 7.1429 / 20.6436 / 0.0 / 57.4572 / 0.0 / 88.9600
- c_qmap: 10.0775 / 14.2857 / 22.2135 / 0.0 / 62.3472 / 4.7619 / 90.8480
- c_no_visual: 9.3023 / 7.1429 / 20.7221 / 0.0 / 61.6137 / 4.7619 / 90.7200

All paired deltas = ablation minus V1; exclusive counts = ablation-only / V1-only:

| Variant | Group | Delta | 95% CI | Exclusive counts |
|---|---|---:|---|---|
|c_static|overall|-2.6042|[-3.4869,-1.7527]|243/406|
|c_static|yes_no|-1.9200|[-2.9478,-0.9328]|101/161|
|c_static|free_form|-3.2865|[-4.6401,-1.9595]|142/245|
|c_static|what|-1.4129|[-2.8298,0.0000]|120/156|
|c_static|where|-14.9144|[-19.5122,-10.4878]|19/80|
|c_static|how|-3.8760|[-10.3704,1.6670]|3/8|
|c_static|other|-7.1429|[-23.0769,0.0000]|0/1|
|c_static|when|0.0000|[0.0000,0.0000]|0/0|
|c_static|why|0.0000|[0.0000,0.0000]|0/0|
|c_qmap|overall|-0.6071|[-1.4307,0.1939]|281/319|
|c_qmap|yes_no|-0.0320|[-0.9814,0.9300]|114/115|
|c_qmap|free_form|-1.1806|[-2.5160,0.1566]|167/204|
|c_qmap|what|+0.1570|[-1.1712,1.5103]|137/133|
|c_qmap|where|-10.0244|[-14.5985,-5.3922]|26/67|
|c_qmap|how|-0.7752|[-5.5118,3.6041]|3/4|
|c_qmap|other|0.0000|[0.0000,0.0000]|0/0|
|c_qmap|when|0.0000|[0.0000,0.0000]|0/0|
|c_qmap|why|+4.7619|[0.0000,15.0000]|1/0|
|c_no_visual|overall|-1.3580|[-2.1390,-0.5650]|251/336|
|c_no_visual|yes_no|-0.1600|[-1.1034,0.7676]|112/117|
|c_no_visual|free_form|-2.5526|[-3.8052,-1.3447]|139/219|
|c_no_visual|what|-1.3344|[-2.6789,0.0000]|115/149|
|c_no_visual|where|-10.7579|[-14.9510,-6.6015]|20/64|
|c_no_visual|how|-1.5504|[-7.6337,3.4783]|3/5|
|c_no_visual|other|-7.1429|[-23.0769,0.0000]|0/1|
|c_no_visual|when|0.0000|[0.0000,0.0000]|0/0|
|c_no_visual|why|+4.7619|[0.0000,15.0000]|1/0|

`question_type:yes/no` duplicates the respective yes_no row exactly. Interpretation: full dynamic branch and Visual18 removal have negative overall paired intervals; qmap overall/Free-form intervals cross zero, while exploratory where interval is negative (41 net losses there vs38 overall). Learned fixed queries still produce image-dependent maps, and question dependence through layer gating remains. Thus no proof that all question conditioning is unnecessary or that original maps localize correctly. Visual18 removes only18432 parameters (~0.99% of V1) but costs1.3580 OA, mostly Free-form (80 of85 net errors); counts do not establish independently additive module effects. Paired intervals capture evaluation uncertainty, not training-seed variance. No new experiments scheduled; uniform remains deferred.

### 2026-10-01 - PathVQA full-attention LoRA-r2 b1a32 completed

- Experiment `pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44`; PathVQA model seed44/data seed42, normalized from-scratch five epochs, fixed epoch5 Validation. Current local launcher verifies standard rank2/alpha4/dropout0.05,192 targets (visual24 qkv/proj + language36 q/k/v/o), expected1,769,472 trainable parameters, LR1e-4/3% warmup/linear, batch1 accumulation32. Original batch2/accumulation16 run failed OOM and remains a separate failed record. Server train_report/commit/runtime/memory not included in returned excerpt.
- Returned Overall57.1018, Yes/No88.8000, Free-form25.4946; how8.5271,other14.2857,what19.9765,when0.0000,where67.4817,why4.7619,yes/no88.8000. Overall95% image-cluster CI[55.5761,58.5195],2000 iterations,bootstrap seed42,832clusters.
- Output `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261001_1`. User supplied paired-statistics location `paired_vs_v1_5ep.json` under this root, not its contents; file not present in local checkout. No paired significance inferred.
- Against original V1 seed44 five-epoch59.3386/90.8800/27.8877: V1 minus LoRA Overall+2.2368, Yes/No+2.0800, Free-form+2.3931; what+2.0800,where+4.8899. V1 parameters1,864,963 vsLoRA1,769,472 (LoRA5.12% fewer than V1): approximately matched, not identical. Both effective batch32 and five epochs, but V1 usesmicrobatch2/accum16 vsLoRA1/32. Matching effective batch does not prove identical optimization; report microbatch difference. One-seed evidence supports V1 advantage under these tested configurations, not universal LoRA superiority/inferiority, paired significance or multi-seed stability.
- Status: completed, no rerun/new GPU work scheduled. Preserve failed prior run and current independent identity.

### 2026-10-01 - LoRA-r2 paired evidence received (supplement to preceding run)

Correction to preceding pending-statistics status: paired report now received for `pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44` vs `pathvqa_v1_norm_fixed_5ep_seed44`. Original JSON direction is LoRA minus V1. Reversing direction, V1 gain Overall+2.236779038185013, image-cluster95%CI[1.2824060634171783,3.2133069013568267]; Yes/No+2.0800 CI[1.0069225928256764,3.157276310487256]; Free-form+2.393107849393747 CI[0.8326933307855736,3.998762297683275]; what+2.0800627943485104 CI[0.431030261896079,3.7037730354840055]; where+4.889975550122244 CI[0.24509803921568626,9.535452322738386]. Bootstrap image_id10000/seed42; overall6259 questions832clusters. V1-only462 vsLoRA-only322, net140, comprising65 Yes/No and75 Free-form. Primary overall interval supports a positive difference between these fixed checkpoints; subgroups are exploratory, not multiplicity-adjusted confirmatory claims. This does not cover training-seed uncertainty or remove the microbatch difference. No new training proposed. All returned details including evaluation/timing and ancillary question-level McNemar statistics preserved below; clustered intervals are the main evidence.

<details><summary>User-supplied paired report excerpt (verbatim; final outer JSON brace absent in attachment)</summary>

```text
{
  "experiment": "pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44",
  "baseline": "pathvqa_v1_norm_fixed_5ep_seed44",
  "bootstrap": {
    "unit": "image_id",
    "iterations": 10000,
    "seed": 42
  },
  "validation_summary": {
    "count": 6259,
    "overall_accuracy": 57.1018,
    "yes_no_accuracy": 88.8,
    "free_form_accuracy": 25.4946,
    "per_answer_type_accuracy": {
      "free-form": 25.4946,
      "yes/no": 88.8
    },
    "per_question_type_accuracy": {
      "how": 8.5271,
      "other": 14.2857,
      "what": 19.9765,
      "when": 0.0,
      "where": 67.4817,
      "why": 4.7619,
      "yes/no": 88.8
    },
    "free_form_question_type_macro_accuracy": 19.1722,
    "clustered_bootstrap": {
      "point_estimate": 57.1018,
      "confidence_level": 0.95,
      "lower": 55.5761,
      "upper": 58.5195,
      "iterations": 2000,
      "seed": 42,
      "image_clusters": 832
    },
    "backend": "lora",
    "base_model": "/root/autodl-tmp/model",
    "checkpoint": "/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261001_1/checkpoints/epoch_5",
    "v0_intervention": "normal",
    "v0_question_mask_policy": null,
    "question_source_policy": null,
    "dynamic_prompt_component_checkpoint": null,
    "dynamic_prompt_components": [],
    "data_root": "/root/autodl-tmp/dataset/pathVQA",
    "split": "validation",
    "max_new_tokens": 32,
    "temperature": 0.0,
    "answer_mode": "raw",
    "instruction": "short-answer",
    "partial_evaluation": false,
    "timing": {
      "methodology": {
        "warmup_runs_excluded": 3,
        "ttft": "generate start to first generated-token logits ready",
        "tpot": "token-count-weighted interval between later token logits",
        "request": "model interface call including preprocessing and decoding",
        "timing_methods": {
          "cuda-events-logits-ready-v2": 6259
        }
      },
      "successful_requests": 6259,
      "model_timed_requests": 6259,
      "generated_tokens": 17979,
      "subsequent_tokens": 11720,
      "ttft_seconds": {
        "count": 6259,
        "mean": 0.056907,
        "p50": 0.055597,
        "p95": 0.058649,
        "min": 0.047965,
        "max": 0.468231
      },
      "tpot_per_request_seconds": {
        "count": 6259,
        "mean": 0.02987,
        "p50": 0.029573,
        "p95": 0.031485,
        "min": 0.029013,
        "max": 0.033435
      },
      "tpot_weighted_seconds": 0.029841,
      "decode_tokens_per_second": 33.511,
      "generation_seconds": {
        "count": 6259,
        "mean": 0.112971,
        "p50": 0.087671,
        "p95": 0.184068,
        "min": 0.077325,
        "max": 0.540737
      },
      "request_seconds": {
        "count": 6259,
        "mean": 0.127375,
        "p50": 0.104776,
        "p95": 0.202106,
        "min": 0.082484,
        "max": 0.559724
      },
      "model_generated_tokens_per_second": 25.427,
      "end_to_end_generated_tokens_per_second": 22.552
    }
  },
  "groups": {
    "overall": {
      "count": 6259,
      "baseline_accuracy": 59.33855248442243,
      "variant_accuracy": 57.10177344623742,
      "delta": -2.236779038185013,
      "variant_only_correct": 322,
      "baseline_only_correct": 462,
      "mcnemar_exact_p": 6.460871178067552e-07,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": -3.2133069013568267,
        "upper": -1.2824060634171783,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 832
      },
      "question_count": 6259,
      "image_clusters": 832,
      "exploratory": false
    },
    "yes_no": {
      "count": 3125,
      "baseline_accuracy": 90.88,
      "variant_accuracy": 88.8,
      "delta": -2.0799999999999983,
      "variant_only_correct": 117,
      "baseline_only_correct": 182,
      "mcnemar_exact_p": 0.0002035765271176517,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": -3.157276310487256,
        "upper": -1.0069225928256764,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 810
      },
      "question_count": 3125,
      "image_clusters": 810,
      "exploratory": true
    },
    "free_form": {
      "count": 3134,
      "baseline_accuracy": 27.887683471601786,
      "variant_accuracy": 25.49457562220804,
      "delta": -2.393107849393747,
      "variant_only_correct": 205,
      "baseline_only_correct": 280,
      "mcnemar_exact_p": 0.0007622293399067396,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": -3.998762297683275,
        "upper": -0.8326933307855736,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 821
      },
      "question_count": 3134,
      "image_clusters": 821,
      "exploratory": true
    },
    "what": {
      "count": 2548,
      "baseline_accuracy": 22.05651491365777,
      "variant_accuracy": 19.97645211930926,
      "delta": -2.0800627943485104,
      "variant_only_correct": 166,
      "baseline_only_correct": 219,
      "mcnemar_exact_p": 0.007962813205220133,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": -3.7037730354840055,
        "upper": -0.431030261896079,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 817
      },
      "question_count": 2548,
      "image_clusters": 817,
      "exploratory": true
    },
    "where": {
      "count": 409,
      "baseline_accuracy": 72.37163814180929,
      "variant_accuracy": 67.48166259168704,
      "delta": -4.889975550122244,
      "variant_only_correct": 36,
      "baseline_only_correct": 56,
      "mcnemar_exact_p": 0.047011561644854,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": -9.535452322738386,
        "upper": -0.24509803921568626,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 408
      },
      "question_count": 409,
      "image_clusters": 408,
      "exploratory": true
    },
    "question_type:how": {
      "count": 129,
      "baseline_accuracy": 10.852713178294573,
      "variant_accuracy": 8.527131782945736,
      "delta": -2.325581395348838,
      "variant_only_correct": 2,
      "baseline_only_correct": 5,
      "mcnemar_exact_p": 0.453125,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": -8.148148148148149,
        "upper": 2.4,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 104
      },
      "question_count": 129,
      "image_clusters": 104,
      "exploratory": true
    },
    "question_type:other": {
      "count": 14,
      "baseline_accuracy": 14.285714285714286,
      "variant_accuracy": 14.285714285714286,
      "delta": 0.0,
      "variant_only_correct": 0,
      "baseline_only_correct": 0,
      "mcnemar_exact_p": 1.0,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": 0.0,
        "upper": 0.0,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 8
      },
      "question_count": 14,
      "image_clusters": 8,
      "exploratory": true
    },
    "question_type:when": {
      "count": 13,
      "baseline_accuracy": 0.0,
      "variant_accuracy": 0.0,
      "delta": 0.0,
      "variant_only_correct": 0,
      "baseline_only_correct": 0,
      "mcnemar_exact_p": 1.0,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": 0.0,
        "upper": 0.0,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 12
      },
      "question_count": 13,
      "image_clusters": 12,
      "exploratory": true
    },
    "question_type:why": {
      "count": 21,
      "baseline_accuracy": 0.0,
      "variant_accuracy": 4.761904761904762,
      "delta": 4.761904761904762,
      "variant_only_correct": 1,
      "baseline_only_correct": 0,
      "mcnemar_exact_p": 1.0,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": 0.0,
        "upper": 15.0,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 20
      },
      "question_count": 21,
      "image_clusters": 20,
      "exploratory": true
    },
    "question_type:yes/no": {
      "count": 3125,
      "baseline_accuracy": 90.88,
      "variant_accuracy": 88.8,
      "delta": -2.0799999999999983,
      "variant_only_correct": 117,
      "baseline_only_correct": 182,
      "mcnemar_exact_p": 0.0002035765271176517,
      "clustered_paired_delta_ci": {
        "confidence_level": 0.95,
        "lower": -3.157276310487256,
        "upper": -1.0069225928256764,
        "iterations": 10000,
        "seed": 42,
        "image_clusters": 810
      },
      "question_count": 3125,
      "image_clusters": 810,
      "exploratory": true
    }
  }
```
</details>

### 2026-10-01 - Five-task final Test suite completed

Suite `outputs/five_task_test_suite/suite_20261001_134036_879815`, final_status completed. Three PathVQA tasks evaluate existing normalized V1 five-epoch seed44/45/46 checkpoints on Test only (not new training); reuse:false refers to these evaluation outputs. Controlled action is final split evaluation, no architecture change. RSVQA tasks train static Visual18+P20 and original CoCoOp-style (no Visual18/no question conditioning) from scratch, model44/data42, five epochs, batch4/accum8, fixed epoch5 Test, per prior approved plan. Full user-returned identities/paths, per-type scores, costs, commits and paired diagnostics preserved below.

- PathVQA Test V1 OA59.8006/59.1904/58.3718, mean59.1209 +/- sampleSD0.7169, range1.4288. YN90.8685 +/-0.4402, Free-form27.3260 +/-1.0709. These are final Test results; do not replace prior Validation58.8592 +/-0.4299 or mix the two. Per-seed image clusters858. Fixed-checkpoint evaluation costs not supplied separately; reported training seconds are historical checkpoint metadata, not new suite training.
- RSVQA Test static69,632 params OA84.4362/AA85.7665, CoCoOp873,120 params OA86.1455/AA86.9998; existing V1 OA85.0360/AA86.0812, LoRA OA87.0552/AA87.7162. CoCoOp minus V1 OA+1.1096 CI[0.2599480103979204,1.9890104677769072], whereas AA+0.9186 CI[-0.8172027559063635,2.73161491919151]. Static minus V1 OA-0.5998 CI[-1.5195441367589724,0.38988303508947314], AA-0.3147 CI[-1.6059840159429775,1.0916204130303087]. V1 minus LoRA OA-2.0192 CI[-2.9982010793523886,-1.059576169532187], AA-1.6350 CI[-2.9727912457990078,-0.3457192609781513]. Thus no clear V1 overall advantage over static in this seed, CoCoOp has positive overall advantage with fewer params; no claim V1 is a universally superior parameter-performance tradeoff. LoRA has three epochs vs V1 five, report exposure mismatch.
- Exploratory count interval accuracy: V1 exceeds static by3.1897, trails CoCoOp by2.4432 and LoRA by5.1578, all corresponding supplied paired intervals exclude zero. This does not prove a pooling mechanism, correct localization, or cross-seed robustness. CoCoOp vs LoRA not directly paired in supplied output. No post-Test tuning proposed; uniform deferred. Existing main-table protocol audits/electrical validation remain separate work.

<details><summary>Full returned suite report</summary>

```text
[RESULT_ROOT] outputs/five_task_test_suite/suite_20261001_134036_879815

===== pathvqa_v1_5ep_seed44_epoch5_test =====
Output: /root/autodl-tmp/Qwen3-VL-modify-test/outputs/five_task_test_suite/suite_20261001_134036_879815/pathvqa_v1_5ep_seed44_epoch5_test 复用: False
Overall: 59.8006
Yes/No: 91.3742
Free-form: 28.1799
Question Types: {'how': 11.5108, 'other': 27.7778, 'what': 23.0208, 'when': 16.6667, 'where': 67.9814, 'why': 0.0, 'yes/no': 91.3742}
Image-cluster CI: {'point_estimate': 59.8006, 'confidence_level': 0.95, 'lower': 58.4932, 'upper': 61.2312, 'iterations': 2000, 'seed': 42, 'image_clusters': 858}
Parameters: 1864963 epochs: 5
Training seconds: 10863.0715 Peak GiB: 23.294
Training commit: 2957ce297637d65a8dcdb5a075c1528072497590

===== pathvqa_v1_5ep_seed45_epoch5_test =====
Output: /root/autodl-tmp/Qwen3-VL-modify-test/outputs/five_task_test_suite/suite_20261001_134036_879815/pathvqa_v1_5ep_seed45_epoch5_test 复用: False
Overall: 59.1904
Yes/No: 90.6603
Free-form: 27.6735
Question Types: {'how': 13.6691, 'other': 27.7778, 'what': 22.4735, 'when': 0.0, 'where': 66.8213, 'why': 4.5455, 'yes/no': 90.6603}
Image-cluster CI: {'point_estimate': 59.1904, 'confidence_level': 0.95, 'lower': 57.785, 'upper': 60.5836, 'iterations': 2000, 'seed': 42, 'image_clusters': 858}
Parameters: 1864963 epochs: 5
Training seconds: 10667.1862 Peak GiB: 23.294
Training commit: f3b9c3922ec44f3388a43b6fc9cf50d28bf30672

===== pathvqa_v1_5ep_seed46_epoch5_test =====
Output: /root/autodl-tmp/Qwen3-VL-modify-test/outputs/five_task_test_suite/suite_20261001_134036_879815/pathvqa_v1_5ep_seed46_epoch5_test 复用: False
Overall: 58.3718
Yes/No: 90.5711
Free-form: 26.1245
Question Types: {'how': 10.7914, 'other': 27.7778, 'what': 21.3426, 'when': 0.0, 'where': 62.877, 'why': 4.5455, 'yes/no': 90.5711}
Image-cluster CI: {'point_estimate': 58.3718, 'confidence_level': 0.95, 'lower': 57.0134, 'upper': 59.8112, 'iterations': 2000, 'seed': 42, 'image_clusters': 858}
Parameters: 1864963 epochs: 5
Training seconds: 10857.1599 Peak GiB: 23.294
Training commit: 548335ffa98594b478117139c80a5ff128cff8ce

===== rsvqa_static_visual18_p20_norm_fixed_5ep_b4a8_seed44 =====
Output: /root/autodl-tmp/Qwen3-VL-modify-test/outputs/five_task_test_suite/suite_20261001_134036_879815/rsvqa_static_visual18_p20_norm_fixed_5ep_b4a8_seed44 复用: False
Overall: 84.4362
AA: 85.7665
Question Types: {'rural_urban': 93.0, 'presence': 91.5059, 'count': 65.9315, 'comp': 92.6287}
Image-cluster CI: {'iterations': 10000, 'seed': 42, 'clusters': 100, 'lower': 83.3966, 'upper': 85.4632}
Parameters: 69632 epochs: 5
Training seconds: 10417.5445 Peak GiB: 11.949
Training commit: 8e4816c69298f158fb297005787fe87d5f544e37

--- 新基线 vs RSVQA V1 （差值=variant−baseline）---
overall: 85.0360 → 84.4362, delta=-0.5998, CI=[-1.5195441367589724, 0.38988303508947314], 独占正确 variant/baseline=413/473
rural_urban: 92.0000 → 93.0000, delta=+1.0000, CI=[-3.0, 5.0], 独占正确 variant/baseline=3/2
presence: 91.0998 → 91.5059, delta=+0.4061, CI=[-0.601619373531827, 1.5311688887324508], 独占正确 variant/baseline=88/76
count: 69.1211 → 65.9315, delta=-3.1897, CI=[-5.4267790609409134, -0.8532132396220845], 独占正确 variant/baseline=230/324
comp: 92.1039 → 92.6287, delta=+0.5247, CI=[-0.40110554777895985, 1.4646853726128075], 独占正确 variant/baseline=92/71
AA delta=-0.3147, CI=[-1.6059840159429775, 1.0916204130303087]

===== rsvqa_cocoop_style_p20_h160_norm_fixed_5ep_b4a8_seed44 =====
Output: /root/autodl-tmp/Qwen3-VL-modify-test/outputs/five_task_test_suite/suite_20261001_134036_879815/rsvqa_cocoop_style_p20_h160_norm_fixed_5ep_b4a8_seed44 复用: False
Overall: 86.1455
AA: 86.9998
Question Types: {'rural_urban': 92.0, 'presence': 92.0812, 'count': 71.5643, 'comp': 92.3538}
Image-cluster CI: {'iterations': 10000, 'seed': 42, 'clusters': 100, 'lower': 85.053, 'upper': 87.2203}
Parameters: 873120 epochs: 5
Training seconds: 10233.2754 Peak GiB: 11.752
Training commit: 8e4816c69298f158fb297005787fe87d5f544e37

--- 新基线 vs RSVQA V1 （差值=variant−baseline）---
overall: 85.0360 → 86.1455, delta=+1.1096, CI=[0.2599480103979204, 1.9890104677769072], 独占正确 variant/baseline=458/347
rural_urban: 92.0000 → 92.0000, delta=+0.0000, CI=[-6.0, 6.0], 独占正确 variant/baseline=5/5
presence: 91.0998 → 92.0812, delta=+0.9814, CI=[0.0, 2.0770693278726067], 独占正确 variant/baseline=92/63
count: 69.1211 → 71.5643, delta=+2.4432, CI=[0.3443466922241245, 4.533233025896379], 独占正确 variant/baseline=278/206
comp: 92.1039 → 92.3538, delta=+0.2499, CI=[-0.5441537489797639, 1.0283625470601063], 独占正确 variant/baseline=83/73
AA delta=+0.9186, CI=[-0.8172027559063635, 2.73161491919151]

===== PathVQA Test 三seed均值 ± 样本标准差 =====
Overall: [59.8006, 59.1904, 58.3718] → 59.1209 ± 0.7169
Yes/No: [91.3742, 90.6603, 90.5711] → 90.8685 ± 0.4402
Free-form: [28.1799, 27.6735, 26.1245] → 27.3260 ± 1.0709
how: 11.9904 ± 1.4976
other: 27.7778 ± 0.0000
what: 22.2790 ± 0.8558
when: 5.5556 ± 9.6225
where: 65.8932 ± 2.6758
why: 3.0303 ± 2.6243
yes/no: 90.8685 ± 0.4402

--- RSVQA V1 − LoRA （差值=variant−baseline）---
overall: 87.0552 → 85.0360, delta=-2.0192, CI=[-2.9982010793523886, -1.059576169532187], 独占正确 variant/baseline=324/526
rural_urban: 92.0000 → 92.0000, delta=+0.0000, CI=[-4.0, 4.0], 独占正确 variant/baseline=2/2
presence: 91.6074 → 91.0998, delta=-0.5076, CI=[-1.5078861306451405, 0.4399360412964955], 独占正确 variant/baseline=79/94
count: 74.2789 → 69.1211, delta=-5.1578, CI=[-7.332671665187308, -3.0128129514989888], 独占正确 variant/baseline=163/315
comp: 92.9785 → 92.1039, delta=-0.8746, CI=[-1.9254874165698654, 0.2011591810141015], 独占正确 variant/baseline=80/115
AA delta=-1.6350, CI=[-2.9727912457990078, -0.3457192609781513]
预算说明: V1 five epochs versus LoRA three epochs, not equal exposure

[FINAL_STATUS] five_task_suite completed
```
</details>


### 2026-10-08 - Electrical final V1 result and LoRA-r8 protocol supplement

- User clarification: private electrical Full-Attention LoRA-r8 result677/954=70.9643605870% (70.96%) used3 epochs and model seed44. This resolves previously unspecified epoch/seed in the current summary; score and denominator unchanged. Exact LoRA experiment name/output path/normalization audit remain unprovided. Preserve prior records and use this explicit correction.
- Completed final V1 electrical evaluation, user-reported675 correct of954 valid samples, accuracy70.7547169811% (70.75%). User confirms the PathVQA V1 configuration transferred completely without changes or electrical-specific tuning. Associated method budget1,864,963 trainable parameters and five-epoch configuration follow the frozen PathVQA V1 definition; actual electrical train_report/parameter audit and evaluation checkpoint not supplied. Model seed, data-seed attestation, exact experiment name, output path, executing commit, split identity, timing/memory, breakdowns, prediction file and paired CI not supplied. Do not assign prior proposed seed47 as observed, and do not fabricate an experiment name or output path.
- Controlled transfer: frozen normalized five-epoch V1/P20/Visual18/two visual LRs/5-11-17 selection/shared offset, as confirmed by user at configuration level; only dataset changes. Fixed private effective denominator954 consistent with corrected historical electrical scoring; no missing-image automatic credits included in reported count.
- Descriptive exact count comparisons, V1 minus historical methods: vsstatic663/954 +12 correct (+1.2578616352pp), vsCoCoOp671/954 +4 (+0.4192872117pp), vsLoRA677/954 -2 (-0.2096436059pp), vsoldQDPT681/954 -6 (-0.6289308176pp). Near point estimates do not establish statistical equivalence/noninferiority; methods differ in training seed/epochs and complete protocol compatibility is unaudited. No paired significance or stable cross-seed benefit claimed. V1 parameters76.1061% fewer than oldQDPT and73.6509% fewer thanLoRA based on fixed method budgets.
- Status: electrical V1 score now received, replacing pending-score status only. Metadata remains pending; no rerun or new GPU experiment scheduled.


### 2026-10-08 - Explicit electrical seed correction and paper result scope

- User corrects preceding electrical LoRA-r8 seed44 assertion after checking its training CFG: `model_path=/root/autodl-tmp/model`, `work_dir=./runs/lora`, `seed=47`, `max_length=1024`, `data=build_training_data_config()`. Correct electrical LoRA-r8 protocol is3 epochs/model seed47. Its677/954=70.96% remains unchanged. Preceding seed44 entry is superseded, retained as historical correction trail.
- User explicitly confirms electrical final V1 also uses model seed47. V1 result675/954=70.75%, unchanged PathVQA configuration and five-epoch budget unchanged. Seed is no longer pending; exact experiment name/output path/train report and paired predictions remain unprovided. StaticP20 andCoCoOp electrical comparisons are likewise seed47. This resolves the reported model-seed mismatch, but epochs differ across methods and numerical proximity does not prove equivalence.
- Paper scope decision: remove all oldQDPT method results/comparisons/development-history material from paper-facing summary, including PathVQA/SLAKE/electrical tables and retrospective parameter/stability comparisons. Updated EXPERIMENT_SUMMARY_20261008.md is paper-facing and has no oldQDPT rows or history appendix. Primary historical ledger remains intact as provenance, not as material to include in the manuscript. Any future manuscript table should use V1, base, static, CoCoOp andLoRA plus applicable final-method ablations; do not reuse retired method names or scores. No new model execution or experiment scheduled.
# 2026-10-08 - SLAKE original CoCoOp seed44 preparation and read-only historical audit

- New experiment `slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44`: code prepared, **not run**. Frozen original CoCoOp P20/H160, no Visual18/question input, strict 873,120 = 51,200 P20 + 821,920 Meta-Net. Model44/data42, from scratch three epochs, fixed epoch3 full bilingual official Test; batch2/accum16/max_length2048/workers2, LR0.3/3e-4, original AdamW/clip1/bf16/3% warmup/linear. Explicit `model_accepts_loss_kwargs=False`, Accelerate accumulation must be1; no manual division. Independent user-run entry `slake/run_cocoop_style_seed44.sh`, no shutdown/retry/Validation/other seed.
- SSH **CPU-file-only** verification on2026-10-08: exact historical PathVQA run `pathvqa/outputs/cocoop/pathvqa_cocoop_style_p20_h160_seed44_20260909/train_report.json` records experiment without date suffix,873120,P20/H160,LR0.3/0.0003,seed44/data42 and completed epoch3/train runtime5962.3328s. Archived epoch3 `cocoop_prompt_config.json` confirms2560/160/P20,shared image bias,before full chat,no question access. `train.log` confirms same LR,epochs3,WD0,warmup0.03/linear. **Historical runtime versions, run commit, batch/accum/workers override values and loss normalization were not recorded; remain unknown.** Introduction source7b94673 supports requested defaults but is not proof of actual historical runtime/overrides. No full training-protocol equivalence claim.
- Exact SLAKE V1 paired baseline verified: `slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928/eval_test/epoch_5`; full Test2094 questions,1582 correct,Overall75.55/CLOSED82.66/OPEN70.83/KVQA55.43/VQA78.49/EN77.29/ZH73.77. Its report confirms five epochs,44/42,originalsplit18,1,864,963 parameters,loss kwargsFalse/Accelerate1,commit327ad6735b60d0830900afc457cd8698189bf2ca; recorded PyTorch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0. CoCoOp3epochs vs V1fiveepochs must remain disclosed.
- Startup binds these explicit directories, validates historical report/log/config and full train/Test image files, independently recomputes saved baseline scoring and checks archived question/reference identity. Outputs are independent timestamp directories under `slake/outputs/cocoop/`, including runtime/config/group counts,epoch checkpoints,logs/stage exits,predictions,summary and10000/seed42 image-cluster paired95%CI (Overall and exploratory OPEN/CLOSED/KVQA/VQA/EN/ZH). Server output `ledger_fragment.md` is returned for immediate Windows-ledger append on result handoff; no server source/ledger writes. No new score exists yet, and no GPU operation was started by the assistant.
- **2026-10-08 execution correction:** `slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44` training completed successfully (924/924, train subprocess exit0, epoch3); the orchestration then failed before Test at `Training subset/checkpoints mismatch`. Exact output `slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_125232_604532`, training commit `6d4ab840a8ae0651229406b76e19eeeb93b6d000`. SSH CPU-only inspection confirmed checkpoints epoch1/2/3 and final all present, saved_epochs=[1,2,3],873120 params,44/42,normalizationFalse/Accelerate1,batch2/accum16,PyTorch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0. Training runtime4407.9755s,loss0.2279439370,peak allocated24050720256 bytes. Root cause: launcher compared raw manifest9835 entries against dataset9834 usable training samples; existing SLAKE loader correctly excludes empty-answer qid1622 (`xmlab281/source.jpg`, English, OPEN/vqa, 'Does the picture contain liver?', answer=''). This is a launcher audit-count error, not missing checkpoints or a training crash. Test has not run; Overall/breakdowns/pairedCI remain unavailable. Original logs/checkpoints preserved. No model/GPU operation or automatic retry was performed by assistant; any evaluation continuation requires user execution. No ledger-only commit.
- **2026-10-08 execution correction:** `slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44` training completed successfully (924/924, train subprocess exit0, epoch3); the orchestration then failed before Test at `Training subset/checkpoints mismatch`. Exact output `slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_125232_604532`, training commit `6d4ab840a8ae0651229406b76e19eeeb93b6d000`. SSH CPU-only inspection confirmed checkpoints epoch1/2/3 and final all present, saved_epochs=[1,2,3],873120 params,44/42,normalizationFalse/Accelerate1,batch2/accum16,PyTorch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0. Training runtime4407.9755s,loss0.2279439370,peak allocated24050720256 bytes. Root cause: launcher compared raw manifest9835 entries against dataset9834 usable training samples; existing SLAKE loader correctly excludes empty-answer qid1622 (`xmlab281/source.jpg`, English, OPEN/vqa, 'Does the picture contain liver?', answer=''). This is a launcher audit-count error, not missing checkpoints or a training crash. Test has not run; Overall/breakdowns/pairedCI remain unavailable. Original logs/checkpoints preserved. No model/GPU operation or automatic retry was performed by assistant; any evaluation continuation requires user execution. No ledger-only commit.


### 2026-10-08 - SLAKE CoCoOp-style seed44 Test Overall returned

- User reports `Overall Accuracy:77.03` for original CoCoOp-style seed44, associated by active task with `slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44`. Intended/previously verified training:3 epochs,873120 parameters,44/42,batch2/accum16,corrected accumulation. Existing verified training root `slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_125232_604532`,commit6d4ab840a8ae0651229406b76e19eeeb93b6d000; this score follows earlier post-training count-audit failure, but actual evaluation root/full summary and completion audit were not returned in this message. Do not infer a new retraining or fabricate Test breakdown/CI.
- Test Overall77.03 (two-decimal displayed precision), other Test scores and paired image-cluster CI unprovided. Against same-seed final V1 Test75.55, point differenceCoCoOp−V1 +1.48pp. CoCoOp3epochs vsV1five remains exposure mismatch. Against V1three-seed mean76.7433 the difference+0.2867 is a single seed versus a mean, not a matched multi-seed method comparison. CoCoOp45/46 not run/reported; no stability conclusion.
- Code-inspected mechanism comparison: both output one sample-conditioned vector broadcast over20 learned static prefix positions. CoCoOp feeds native merger visual-token mean through2560->160->2560 ReLU; V1 usesVisual18, question-context pooling,question-conditioned5/11/17 maps pooling the same final merger values, per-layerLN/2560->64 projections,questionsoftmax gates,concat192/ReLU/192->2560. CoCoOp is a simplified related architecture but not a strict identical-init subset: native value features/LN/query dependence/initializations/learning rates differ. Failure of a more complex auxiliary selector to improve task score is consistent with redundant task conditioning or information loss/optimization costs, not proof of these hypotheses. Corrected V1'sPathVQAC-qmap totalCI crosseszero; independent question selection cannot be described as universally necessary.
- Status: CoCoOp seed44 now has a returned score, overriding earlier no-Test-score statements. Pairing/breakdowns pending; no tuning or new seeds automatically scheduled.


### 2026-10-08 - SLAKE CoCoOp seed44 complete Test summary and evaluation identity

- Supplemental result for `slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44`; this completes the prior partial score entry. Frozen original CoCoOp873120 parameters,3 epochs,model44/data42,b2a16 and verified normalization as earlier training record. Evaluation root `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_143843_694315`; predictions `eval_test/epoch_3/slake_predictions.json`,summary `eval_test/epoch_3/slake_summary.json`. Actual loaded checkpoint is the earlier completed training root `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_125232_604532/checkpoints/epoch_3`. This is evaluation continuation, not a new training run; preserve original post-training audit failure.
- Official Test metric `slake_vqa_normalized_exact_match`,2094 questions,1613 correct,481 wrong,missing_predictions0/extra_predictions0,partial_evaluationFalse. All-language/no base_types restriction; questions `/root/autodl-tmp/dataset/slake/test.json`,images `/root/autodl-tmp/dataset/slake/imgs`,base model `/root/autodl-tmp/model`,backendcocoop-style,max_new_tokens32,temperature0,raw answers,language-aware-short-answer instruction.
- Returned Test Overall77.03; CLOSED83.85/OPEN72.50; KVQA59.93/VQA79.53; EN78.04/ZH75.99. Against verified V1 seed44 five-epoch2094/1582 correct: CoCoOp has31 net additional correct (+1.4804202483pp exact count; displayed+1.48). Displayed subgroup gains CLOSED+1.19,OPEN+1.67,KVQA+4.50,VQA+1.04,EN+0.75,ZH+2.22pp. All point estimates favor CoCoOp, not an observed binary/open or language tradeoff. KVQA/VQA are question partitions, not clean causal knowledge/vision isolation. No paired exclusive counts or paired/standalone scoreCI supplied; no subgroup significance or cross-seed ranking inferred. CoCoOp45/46 not scheduled here.
- Timing methodology:3 excluded warmups,all2094 usecuda-events-logits-ready-v2. TTFT count2094 mean0.063735/p500.040866/p950.109294/min0.032755/max0.327885s. TPOT per-request count2094 mean0.019797/p500.019373/p950.021923/min0.018724/max0.030253s;weighted0.019765s/token,decode50.595tokens/s. Generation count2094 mean0.098269/p500.079659/p950.166027/min0.051668/max0.349295s; request count2094 mean0.112944/p500.086680/p950.196612/min0.056552/max0.377432s. Returned generated_tokens47612,subsequent_tokens3638,model_generated_tokens_per_second231.379,end_to_end_generated_tokens_per_second201.315. Counting audit note:47612-(2094+3638)=41880=20*2094; reported aggregate generated count is consistent with including all20 prefix tokens/request and must not be used as a clean output-token throughput without source verification. Accuracy counts are separate from timing count issue. No inference speed comparison with V1 made.
- Result interpretation: original simpler image-conditioned prefix baseline performs better than tested V1seed44 in all returned partitions with fewer params and fewer epochs. One-seed result does not establish universal superiority or cause; no Test-driven tuning/retraining proposed. Final paired-statistics file contents remain to be handed back.
# 2026-10-08 - Independent SLAKE V10 weighted-map H160 preparation (not trained)

- Exact planned experiment `slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44`, SLAKE full bilingual Train/Test, model44/data42. Controlled head change: keep original V1 P20/Visual18/code17 S8+Av10/question encoder/QK maps5,11,17/question layer_gate. Sum each block-major group of4 patch probabilities onto the corresponding native merger token, then `a=sum(beta_l*merged_map_l)`, `z=sum(a_j*V_native,j)`, bias-enabled2560->160/ReLU/160->2560 Meta-Net, one shared offset onP20 before full chat. No repeated softmax/divide3, no per-layer Value LN/projector or192->2560 head in final graph. Independent files/method/config/weights/backend; original V1 remains unchanged.
- Exact CPU mock module counts: P20 51200/S8 8192/Av10 10240/question_context345088/maps448896/layer_condition387/meta_net821920 = **1,685,923**. Parent V1 is constructed only to retain exact common initialization, then all value_norms/value_blocks/prefix_output are removed. CPU tests compare28 retained tensors against fresh same-seed V1 and verify RNG unchanged; new head initialized under independent CPU generator state. First Linear uses its regular default init; outputNormal(0,1e-4),zero bias, measured mock std0.0000998816. New head LR1e-4, not CoCoOp3e-4; no pretrained checkpoint initialization.
- Five epochs from scratch,save3/4/5,fixedepoch5 Test only,9834 existing-loader effective Train (empty-answerqid1622 excluded and logged)/2094 Test. Batch2/accum16,max_length2048,workers2,bf16,AdamW(.9,.999)/eps1e-8/WD0/clip1,warmup0.03/linear toepoch5. RatesP20.3/S8 3e-5/Av10 1e-4/all other groups1e-4. Trainer loss kwargsFalse/Accelerate1; no manual division. Startup captures versions,commit,group budget,preclip first-batch gradients/postclip group gradients/Trainer preclip grad_norm in state,three map entropies/layer weights/summary and offset RMS/offset-P20 ratio,training runtime/peak memory and independent epochs. Train-only real-batch/greedy roundtrip/cache/one vision+LLM checks run only under user's GPU launch,restore RNG before training.
- Precisely bound baselines: main CoCoOp eval `slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_143843_694315/eval_test/epoch_3` (77.03,1613/2094),weights explicitly from trainingrun`..._20261008_125232_604532/checkpoints/epoch_3`; reference V1 `slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928/eval_test/epoch_5` (75.55,1582/2094). CPU-only binding/recomputed scoring/reference identity checks passed. V10fiveepochs/Visual18 vs CoCoOp3epochs/noVisual18 and different Meta-Net LR/init: do not attribute all differences solely to question maps. Overall and exploratory OPEN/CLOSED/KVQA/VQA/EN/ZH paired img_name-cluster95%CI10000/seed42 will use existing baseline predictions, not regenerate them.
- Checks: local4 NumPy/AST tests+Python compile,remote2 CPU tensor mock tests with GPU completely invisible/in-memory local source only,plusCPU artifact precheck. Precheck path `slake/outputs/v10/prechecks/slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44_20261008_173524_696069` is **not a training result**. Actual experiment output under`slake/outputs/v10/runs/`; entry`slake/run_v10_weighted_map_metanet_seed44.sh`,fail-stop/no retries/no shutdown. No real model loading/GPU run/training/Test performed by assistant. Scores and real training costs currently unavailable; successful/failed result fragments await user handoff for immediate Windows ledger update, not server-source edits or ledger-only commits.

### 2026-10-08 - SLAKE V10 seed44 fixed epoch5 Test completed (user-returned)

- Exact experiment: slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44, SLAKE full bilingual official Test, fixed epoch5 checkpoint. Model seed44/data seed42 and five-epoch from-scratch V1 protocol are defined by the implemented plan/launcher; the user returned completed evaluation and paired comparisons, not the actual train_report or training diagnostic trajectory. This completion supersedes earlier not-trained/not-evaluated preparation status without removing history.
- Controlled head change versus V1: preserve P20, code17 Visual18 S8+Av10, question encoder, code5/11/17 Q/K maps, question-conditioned softmax layer weights and native final merger Values. Map fusion is a=sum_l beta_l*merged_map_l, z=sum_j a_j*V_j, with no resoftmax/divide3. Replace three Value LayerNorms/2560->64 projections/weighted concat/192->2560 output head with one bias-enabled2560->160->ReLU->2560 Meta-Net, shared offset on all20 P20 positions. Common28 initial tensors and total1,685,923 parameters were previously CPU-verified; the actual GPU training report has not been handed back. Relative to V1 1,864,963, implementation budget is179,040 fewer parameters (-9.6001904595%). This is a joint change of head organization, normalization, bottleneck allocation/width and biases, not a single-factor test of64-dimensional overlap or gradient suppression.
- Planned/audited source training configuration: fiveepochs,save3/4/5,model44/data42,batch2/accum16,max_length2048,workers2,bf16,AdamW(.9,.999)/eps1e-8/WD0/clip1,3%warmup/linear across5epochs. RatesP20.3/S8 3e-5/Av10 1e-4/question context/maps/layer gate/Meta-Net1e-4; outputNormal(0,1e-4),zero bias. Trainer loss kwargsFalse/Accelerate1. Local implementation commit57c31e6c18a4ae608f9f9655f6ea55b2dfb36d19; actual executing commit/runtime versions/train loss/runtime/peak memory/per-step activation or gradient records absent from returned attachment, not inferred from local HEAD.
- Returned Test:2094 questions,1632 correct,462 wrong,0 missing/extra,Overall77.94 (exact77.93696275071633 from paired payload). CLOSED84.69/OPEN73.45;KVQA58.80/VQA80.73;EN78.13/ZH77.73. Official language=all,base_types=[],expected_split=test,partial_evaluation=false,max_new_tokens32,temperature0,raw answers,language-aware-short-answer instruction.
- Output root: /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/v10/runs/slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44_20261008_174753_716357. Checkpoint checkpoints/epoch_5; predictions and summary eval_test/epoch_5/slake_predictions.json and slake_summary.json. Baselines remain explicitly bound to CoCoOp seed44 epoch3 Test77.03 and V1 seed44 epoch5 Test75.55 per preparation record; no baseline rerun or new seed.
- Primary paired results (V10 minus comparator): vsV1 +2.3877745941pp,95%CI[+0.8888733053,+3.8911129095],V10-only135/V1-only85,net50 correct; vsCoCoOp +0.9073543457pp,CI[-0.3791514138,+2.2211720227],100/81,net19. Overall comparison has2094 questions/180 official image clusters. Launcher specifies image img_name-cluster bootstrap10000/seed42; returned intervals and cluster counts are observed, iteration/seed fields are not repeated in payload. All subgroup comparisons below are exploratory.
- Exploratory improvement versus V1: OPEN+2.6232114467pp,CI[+0.7393906926,+4.5342126958],net33 correct;ZH+3.9690222652pp,CI[+1.5136466008,+6.6084379442],net41;EN net9. These are overlapping partitions, not additive explanations of mechanism.
- Timing returned: TTFT mean0.066221/p500.042961/p950.110191s;weightedTPOT0.019881s/token,50.299decode tokens/s;request mean0.116687/p500.088622/p950.203678s. All2094 timed requests,cuda-events-logits-ready-v2,3 excluded warmups. Generated5829/subsequent3735 satisfies5829=2094+3735, unlike the prior CoCoOp prefix-count discrepancy. This arithmetic does not independently validate all timing implementation or support speed comparison across unverified runtime hardware; complete timing data are preserved verbatim below.
- Interpretation: simplifying the retained-map condition head improves this same-seed,five-epoch V1 comparison with a positive overall paired interval and fewer implementation parameters. The result supports the shared Meta-Net direction, not proof of64-dimensional redundancy, blocked gradients, correct localization or multi-seed superiority. CoCoOp is numerically lower but its paired interval includes0; V10 has5epochs/Visual18 vsCoCoOp3epochs/noVisual18 and different LR/init, so no confirmed CoCoOp advantage attributable solely to maps. No further training/evaluation/seed was started or scheduled by the assistant. This task only read existing files and updated local ledgers; no ledger-only commit/push.
| Comparator | Group | N | Images | Delta pp | 95% paired CI | V10-only / comparator-only | Exploratory |
|---|---|---:|---:|---:|---|---:|---|
| v1 | overall | 2094 | 180 | 2.3878 | [0.8889, 3.8911] | 135/85 | False |
| v1 | OPEN | 1258 | 180 | 2.6232 | [0.7394, 4.5342] | 86/53 | True |
| v1 | CLOSED | 836 | 180 | 2.0335 | [-0.1179, 4.1622] | 49/32 | True |
| v1 | kvqa | 267 | 103 | 3.3708 | [-0.7463, 7.8431] | 21/12 | True |
| v1 | vqa | 1827 | 180 | 2.2441 | [0.6622, 3.7879] | 114/73 | True |
| v1 | en | 1061 | 96 | 0.8483 | [-0.8396, 2.6126] | 42/33 | True |
| v1 | zh | 1033 | 96 | 3.9690 | [1.5136, 6.6084] | 93/52 | True |
| cocoop | overall | 2094 | 180 | 0.9074 | [-0.3792, 2.2212] | 100/81 | False |
| cocoop | OPEN | 1258 | 180 | 0.9539 | [-0.6955, 2.5602] | 59/47 | True |
| cocoop | CLOSED | 836 | 180 | 0.8373 | [-1.1436, 2.8674] | 41/34 | True |
| cocoop | kvqa | 267 | 103 | -1.1236 | [-4.9470, 2.7451] | 16/19 | True |
| cocoop | vqa | 1827 | 180 | 1.2042 | [-0.1684, 2.5724] | 84/62 | True |
| cocoop | en | 1061 | 96 | 0.0943 | [-1.5903, 1.7495] | 39/38 | True |
| cocoop | zh | 1033 | 96 | 1.7425 | [-0.2073, 3.7849] | 61/43 | True |

<details>
<summary>Complete user-returned V10 evaluation and comparisons</summary>

~~~~text
========== SLAKE Evaluation ==========
Overall Accuracy: 77.94
Per Answer Type: {'CLOSED': 84.69, 'OPEN': 73.45}
Per Question Type: {'kvqa': 58.8, 'vqa': 80.73}
Per Language: {'en': 78.13, 'zh': 77.73}
TTFT: {'count': 2094, 'mean': 0.066221, 'p50': 0.042961, 'p95': 0.110191, 'min': 0.035977, 'max': 0.304038}
TPOT: 0.019881 s/token, 50.299 token/s
Request Latency: {'count': 2094, 'mean': 0.116687, 'p50': 0.088622, 'p95': 0.203678, 'min': 0.060624, 'max': 0.390968}
Predictions: /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/v10/runs/slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44_20261008_174753_716357/eval_test/epoch_5/slake_predictions.json
Summary: /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/v10/runs/slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44_20261008_174753_716357/eval_test/epoch_5/slake_summary.json
[SLAKE_V10_DONE] {"summary": {"metric": "slake_vqa_normalized_exact_match", "total": 2094, "correct": 1632, "wrong": 462, "missing_predictions": 0, "extra_predictions": 0, "extra_prediction_ids": [], "overall_accuracy": 77.94, "per_answer_type_accuracy": {"CLOSED": 84.69, "OPEN": 73.45}, "per_question_type_accuracy": {"kvqa": 58.8, "vqa": 80.73}, "per_language_accuracy": {"en": 78.13, "zh": 77.73}, "backend": "v10-weighted-map", "base_model": "/root/autodl-tmp/model", "checkpoint": "/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/v10/runs/slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44_20261008_174753_716357/checkpoints/epoch_5", "questions": "/root/autodl-tmp/dataset/slake/test.json", "image_root": "/root/autodl-tmp/dataset/slake/imgs", "language": "all", "base_types": [], "expected_split": "test", "max_new_tokens": 32, "temperature": 0.0, "answer_mode": "raw", "instruction": "language-aware-short-answer", "partial_evaluation": false, "timing": {"methodology": {"warmup_runs_excluded": 3, "ttft": "generate start to first generated-token logits ready", "tpot": "token-count-weighted interval between later token logits", "request": "model interface call including preprocessing and decoding", "timing_methods": {"cuda-events-logits-ready-v2": 2094}}, "successful_requests": 2094, "model_timed_requests": 2094, "generated_tokens": 5829, "subsequent_tokens": 3735, "ttft_seconds": {"count": 2094, "mean": 0.066221, "p50": 0.042961, "p95": 0.110191, "min": 0.035977, "max": 0.304038}, "tpot_per_request_seconds": {"count": 2094, "mean": 0.01992, "p50": 0.019658, "p95": 0.021483, "min": 0.019001, "max": 0.030122}, "tpot_weighted_seconds": 0.019881, "decode_tokens_per_second": 50.299, "generation_seconds": {"count": 2094, "mean": 0.101841, "p50": 0.082052, "p95": 0.170684, "min": 0.055436, "max": 0.384401}, "request_seconds": {"count": 2094, "mean": 0.116687, "p50": 0.088622, "p95": 0.203678, "min": 0.060624, "max": 0.390968}, "model_generated_tokens_per_second": 27.333, "end_to_end_generated_tokens_per_second": 23.856}}, "comparisons": {"cocoop": {"overall": {"count": 2094, "image_clusters": 180, "baseline_accuracy": 77.02960840496657, "variant_accuracy": 77.93696275071633, "delta": 0.9073543457497664, "variant_only_correct": 100, "baseline_only_correct": 81, "ci95": [-0.37915141382340184, 2.22117202268431], "exploratory": false}, "OPEN": {"count": 1258, "image_clusters": 180, "baseline_accuracy": 72.49602543720191, "variant_accuracy": 73.44992050874404, "delta": 0.953895071542135, "variant_only_correct": 59, "baseline_only_correct": 47, "ci95": [-0.695531222095913, 2.5602235881522137], "exploratory": true}, "CLOSED": {"count": 836, "image_clusters": 180, "baseline_accuracy": 83.85167464114832, "variant_accuracy": 84.68899521531101, "delta": 0.837320574162689, "variant_only_correct": 41, "baseline_only_correct": 34, "ci95": [-1.1436305310668349, 2.867383512544803], "exploratory": true}, "kvqa": {"count": 267, "image_clusters": 103, "baseline_accuracy": 59.9250936329588, "variant_accuracy": 58.80149812734082, "delta": -1.1235955056179776, "variant_only_correct": 16, "baseline_only_correct": 19, "ci95": [-4.946996466431095, 2.7450980392156863], "exploratory": true}, "vqa": {"count": 1827, "image_clusters": 180, "baseline_accuracy": 79.52928297755884, "variant_accuracy": 80.73344280240832, "delta": 1.2041598248494836, "variant_only_correct": 84, "baseline_only_correct": 62, "ci95": [-0.16844705978840316, 2.5723868875091123], "exploratory": true}, "en": {"count": 1061, "image_clusters": 96, "baseline_accuracy": 78.03958529688973, "variant_accuracy": 78.13383600377003, "delta": 0.09425070688030246, "variant_only_correct": 39, "baseline_only_correct": 38, "ci95": [-1.5902712815715623, 1.7495395948434622], "exploratory": true}, "zh": {"count": 1033, "image_clusters": 96, "baseline_accuracy": 75.99225556631171, "variant_accuracy": 77.73475314617619, "delta": 1.7424975798644766, "variant_only_correct": 61, "baseline_only_correct": 43, "ci95": [-0.20732463970043968, 3.784899237999458], "exploratory": true}}, "v1": {"overall": {"count": 2094, "image_clusters": 180, "baseline_accuracy": 75.54918815663801, "variant_accuracy": 77.93696275071633, "delta": 2.387774594078323, "variant_only_correct": 135, "baseline_only_correct": 85, "ci95": [0.8888733052828424, 3.8911129095047867], "exploratory": false}, "OPEN": {"count": 1258, "image_clusters": 180, "baseline_accuracy": 70.82670906200318, "variant_accuracy": 73.44992050874404, "delta": 2.623211446740868, "variant_only_correct": 86, "baseline_only_correct": 53, "ci95": [0.7393906925505898, 4.534212695795548], "exploratory": true}, "CLOSED": {"count": 836, "image_clusters": 180, "baseline_accuracy": 82.6555023923445, "variant_accuracy": 84.68899521531101, "delta": 2.033492822966508, "variant_only_correct": 49, "baseline_only_correct": 32, "ci95": [-0.1179280089550244, 4.162217628358444], "exploratory": true}, "kvqa": {"count": 267, "image_clusters": 103, "baseline_accuracy": 55.43071161048689, "variant_accuracy": 58.80149812734082, "delta": 3.370786516853933, "variant_only_correct": 21, "baseline_only_correct": 12, "ci95": [-0.746268656716418, 7.8431372549019605], "exploratory": true}, "vqa": {"count": 1827, "image_clusters": 180, "baseline_accuracy": 78.48932676518884, "variant_accuracy": 80.73344280240832, "delta": 2.2441160372194844, "variant_only_correct": 114, "baseline_only_correct": 73, "ci95": [0.662233401724629, 3.7879099895162494], "exploratory": true}, "en": {"count": 1061, "image_clusters": 96, "baseline_accuracy": 77.28557964184732, "variant_accuracy": 78.13383600377003, "delta": 0.8482563619227079, "variant_only_correct": 42, "baseline_only_correct": 33, "ci95": [-0.8395914702189985, 2.6126409565422146], "exploratory": true}, "zh": {"count": 1033, "image_clusters": 96, "baseline_accuracy": 73.76573088092933, "variant_accuracy": 77.73475314617619, "delta": 3.9690222652468634, "variant_only_correct": 93, "baseline_only_correct": 52, "ci95": [1.5136466007899136, 6.608437944244994], "exploratory": true}}}}
~~~~
</details>
## 2026-10-10 最后一试完成：修正版五轮P20 LR0.3→0.1未获收益

实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_p20lr01_norm_fixed_5ep_seed44`，PathVQA model seed44/data seed42，从零五轮，固定epoch5完整Validation6259题/832图；唯一受控改动P20基础LR0.3→0.1，结构、初始化、其他组LR、batch2/accum16、3%warmup/linear五轮、clip1及归一化沿用修正版。train_report共有30初值核验一致；实际参数1691043，commit b081a5819bb89afc8639808a2f4cbac82fbd01de，训练10737.3951秒，全程平均train_loss0.7041064902825084。完整用户原文（包含全部summary、配对、运行审计、成本）保留于 `pathvqa/outputs/cpu_curve_audit_20261009/v10_p20lr01_user_result.txt`。

Overall58.3160、Yes/No90.4320、Free-form26.2923；how8.5271/other7.1429/what21.1538/when0/where66.7482/why0。独立图像簇95%CI[56.87,59.76]（用户显示精度），832簇。TTFT mean .051249秒，TPOT .01938秒/token（51.6token/s），request mean .103625秒；不作跨运行速度归因。

| 对照 | 分组 | delta(pp) | 95%图像簇配对CI | 本次独占/对照独占 |
| --- | --- | ---: | --- | ---: |
| 修正版五轮58.6835 | Overall | -0.3675 | [-1.193577,0.448215] | 301/324 |
| 修正版五轮 | Yes/No | +0.1920 | [-0.734144,1.106844] | 108/102 |
| 修正版五轮 | Free-form | -0.9253 | [-2.314975,0.437774] | 193/222 |
| 修正版五轮 | what | -0.9419 | [-2.383790,0.515479] | 153/177 |
| 修正版五轮 | where | -0.2445 | [-4.634146,3.921569] | 39/40 |
| V1五轮59.3386 | Overall | -1.0225 | [-1.889886,-0.162205] | 314/378 |
| V1五轮 | Yes/No | -0.4480 | [-1.384417,0.454900] | 110/124 |
| V1五轮 | Free-form | -1.5954 | [-3.066143,-0.162537] | 204/254 |
| V1五轮 | what | -0.9027 | [-2.431251,0.625739] | 177/200 |
| V1五轮 | where | -5.6235 | [-9.803922,-1.466993] | 26/49 |

配对10000次/seed42，Overall6259/832，Yes-No3125/810，Free-form3134/821，what2548/817，where409/408；分项探索性。服务器输出根 `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_p20lr01_norm_fixed_5ep_seed44_20261009_205616_564163`，预测/summary分别 `eval_validation/epoch_5/pathvqa_predictions.json`及`pathvqa_summary.json`。

结论：P20降至0.1没有明确收益；相对修正版五轮总体和全部分项配对CI均跨零，不证明等效或明确退化。对V1总体配对CI低于零。早期P20尺度线索未转化为这一单变量配置的提升，不支持继续此方向盲调，也不排除其他学习率/调度问题。此前“P20过早塑造前缀”仍为未证实假设。最后一试授权完成，冻结V1论文主方案，保留修正版五轮作探索结果；不追加训练/seed/Test/重试。助手仅CPU读取及记录，无GPU操作，无账本独立commit/push。
## 2026-10-10 LR0.1专用代码清理及跨数据集准备（未运行）

P20 LR0.1既有PathVQA结果58.3160完整记录仍保留；用户要求停止该方向并清理代码，本次仅移除专用train/bash文件和调度分支，不删除任何checkpoint、日志、预测或历史条目，Git可恢复代码。新准备SLAKE→RSVQA-LR V10条件头修正版五轮seed44/data42，实验名分别 `slake_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44` 与 `rsvqa_lr_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`；固定epoch5完整Test，不跑Validation。参数1691043，完全沿PathVQA五轮修正版基础LR/初始化/batch2累积16/归一化False/clip1/五轮scheduler，只迁移数据和官方评分。代码准备不代表运行完成，分数未知。独立输出，Bash串行失败即停、日志状态先保存后/usr/bin/shutdown，调用失败日志保存。所有GPU用户执行，结果回传后追加精确实验成绩。
## 2026-10-10 V10条件头修正版SLAKE与RSVQA-LR完成：用户回传汇总

两项均为条件头修正版（融合摘要LN、默认Linear初始化、Meta-Net基础LR3e-4）；实验名记录5ep/seed44。按本次用户上下文为既定五轮seed44配置迁移，未回传train_report/执行commit/分组LR或实测参数，不能把本机配置视为远程实测。相同结构参数预算1,691,043，仅参考此前PathVQA实测；P20是否沿用0.3需报告核实，不将本次与p20lr01混同。未提供成本、CI、配对和逐步诊断。本条仅记录成绩，不安排额外训练。

1. `slake_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`，SLAKE，按既定入口固定epoch5全语言Test（2094题；本次用户未另报count/split元数据）：Overall77.32，CLOSED83.61/OPEN73.13，KVQA62.17/VQA79.53，EN78.04/ZH76.57。用户输出 `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/v10_head_fixed/slake_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261010_002751_59639`。对同seed V1点差+1.77pp，对CoCoOp三轮+0.29pp，对原V10五轮77.94为-0.62pp；不宣称显著性或等效。
2. `rsvqa_lr_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`，RSVQA-LR，按既定入口固定epoch5 Test（10004题/100图；本次用户未另报count/split元数据）：OA85.4558/AA85.7081，rural_urban89.0/presence91.6751/count70.0034/comp92.1539。用户输出 `/root/autodl-tmp/Qwen3-VL-modify-test/RSVQA/outputs/v10_head_fixed/rsvqa_lr_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261010_002751_59639`。对V1 OA+0.4198pp而AA-0.3731pp；对CoCoOp OA-0.6897pp，对LoRA-r8 OA-1.5994pp。AA为题型平均，OA更高不表示各题型均改善。原版V10 RSVQA未回传，不补造。

汇总文件 `EXPERIMENT_COMPARISON_20261010.md` 保留Test/Validation、版本、epoch、seed及历史协议边界；旧QDPT不进入本次表格。两账本/计划即时补录，无仅账本commit/push、无模型/GPU操作。
