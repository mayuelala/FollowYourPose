## <font color="red"> 🕺🕺🕺 Follow Your Pose 💃💃💃 </font>: <br> Pose-Guided Text-to-Video Generation using Pose-Free Videos

[Yue Ma](https://mayuelala.github.io/), [Yingqing He](http://vinthony.github.io/), [Xiaodong Cun](http://vinthony.github.io/), [Xintao Wang](https://xinntao.github.io/), [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ), [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=zh-CN), and [Qifeng Chen](https://cqf.io)

<a href='https://arxiv.org/abs/2303.09535'><img src='https://img.shields.io/badge/ArXiv-2303.09535-red'></a> 
<a href='https://fate-zero-edit.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  [![GitHub](https://img.shields.io/github/stars/mayuelala/FollowYourPose?style=social)](https://github.com/mayuelala/FollowYourPose)


<!-- ![fatezero_demo](./docs/teaser.png) -->

<table class="center">
  <td><img src="gif_results/A Astronaut.gif"></td>
  <td><img src="gif_results/ertA Hulk on the sea .gif"></td>
  <tr>
  <td width=25% style="text-align:center;">"A astronaut, brown background"</td>
  <td width=25% style="text-align:center;">"A Hulk, on the sea"</td>
  <!-- <td width=25% style="text-align:center;">"Wonder Woman, wearing a cowboy hat, is skiing"</td>
  <td width=25% style="text-align:center;">"A man, wearing pink clothes, is skiing at sunset"</td> -->
</tr>
<td><img src="gif_results/a man in the park, Van Gogh style.gif"></td>
<td><img src="gif_results/vervA Stormtrooper on the sea.gif"></td>
<tr>
<td width=25% style="text-align:center;">"A man in the park, Van Gogh style"</td>
<td width=25% style="text-align:center;">"The Stormtroopers, on the beach"</td>
</tr>
</table >

## 💃💃💃 Abstract
<b>TL;DR: We tune 2D stable-diffusion to generate the character videos from pose and text description.</b>

<details><summary>CLICK for full abstract</summary>


> Generating text-editable and pose-controllable character videos have an imperious demand in creating various digital human. Nevertheless, this task has been restricted by the absence of a comprehensive dataset featuring paired video-pose captions and the generative prior models for videos. In this work, we design a novel two-stage training scheme that can utilize easily obtained datasets (i.e., image pose pair and pose-free video) and the pre-trained text-to-image (T2I) model to obtain the pose-controllable character videos. Specifically, in the first stage, only the keypoint-image pairs are used only for a controllable textto-image generation. We learn a zero-initialized convolutional encoder to encode the pose information. In the second stage, we finetune the motion of the above network via a pose-free video dataset by adding the learnable temporal self-attention and reformed cross-frame self-attention blocks. Powered by our new designs, our method successfully generates continuously pose-controllable character videos while keeps the editing and concept composition ability of the pre-trained T2I model. The code and models will be made publicly available.
</details>

## 🕺🕺🕺 Changelog
<!-- A new option store all the attentions in hard disk, which require less ram. -->
- 2023.03.30 Release Paper and Project page!

## Todo

- [ ] Release the code, config and checkpoints for teaser
- [ ] Memory and runtime profiling
- [ ] Hands-on guidance of hyperparameters tuning
- [ ] Colab
- [ ] Release configs for other result and in-the-wild dataset
- [ ] hugging-face: inprogress
- [ ] Release more application


## 💃💃💃 Results with Stable Diffusion
We show results regarding various pose sequences and text prompts.

Note mp4 and gif files in this github page are compressed. 
Please check our [Project Page](https://follow-your-pose.github.io/) for mp4 files of original video results.
<table class="center">

<tr>
  <td><img src="gif_results/A Robot is dancing in Sahara desert.gif"></td>
  <td><img src="gif_results/sdsdA Iron man on the beach.gif"></td>
    <td><img src="gif_results/A Panda on the sea.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A Robot, in Sahara desert"</td>
  <td width=25% style="text-align:center;">"A Iron man, on the beach"</td>
  <td width=25% style="text-align:center;">"A panda, son the sea"</td>
</tr>
<!--#########################################################-->
<tr>
  <td><img src="gif_results/a man in the park, Van Gogh style.gif"></td>
  <td><img src="gif_results/fireman in the beach.gif"></td>
  <td><img src="gif_results/Batman brown background.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"A man in the park, Van Gogh style"</td>
  <td width=25% style="text-align:center;">"The fireman in the beach"</td>
  <td width=25% style="text-align:center;">"Batman, brown background"</td>
</tr>
<!--#########################################################-->


<tr>
  <td><img src="gif_results/ertA Hulk on the sea .gif"></td>
  <td><img src="gif_results/lokA Superman on the forest.gif"></td>
  <td><img src="gif_results/A Iron man on the snow.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"A Hulk, on the sea"</td>
  <td width=25% style="text-align:center;">"A superman, in the forest"</td>
  <td width=25% style="text-align:center;">"A Iron man, in the snow"</td>
</tr>

<!--#########################################################-->
<tr>
  <td><img src="gif_results/A man in the forest, Minecraft.gif"></td>
  <td><img src="gif_results/a man in the sea, at sunset.gif"></td>
  <td><img src="gif_results/James Bond, grey simple background.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"A man in the forest, Minecraft."</td>
  <td width=25% style="text-align:center;">"A man in the sea, at sunset"</td>
  <td width=25% style="text-align:center;">"James Bond, grey simple background"</td>
</tr>
<!--#########################################################-->

<tr>
  <td><img src="gif_results/vryvA Panda on the sea.gif"></td>
  <td><img src="gif_results/nbthA Stormtrooper on the sea.gif"></td>
  <td><img src="gif_results/egbA astronaut on the moon.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"A Panda on the sea."</td>
  <td width=25% style="text-align:center;">"A Stormtrooper on the sea"</td>
  <td width=25% style="text-align:center;">"A astronaut on the moon"</td>
</tr>
<!--#########################################################-->

<tr>
  <td><img src="gif_results/sssA astronaut on the moon.gif"></td>
  <td><img src="gif_results/ccA Robot in Antarctica.gif"></td>
  <td><img src="gif_results/zzzzA Iron man on the beach..gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"A astronaut on the moon."</td>
  <td width=25% style="text-align:center;">"A Robot in Antarctica."</td>
  <td width=25% style="text-align:center;">"A Iron man on the beach."</td>
</tr>

<!--#########################################################-->

<tr>
  <td><img src="gif_results/yrvA Obama in the desert.gif"></td>
  <td><img src="gif_results/dfcA Astronaut on the beach.gif"></td>
  <td><img src="gif_results/sasqA Iron man on the snow.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"The Obama in the desert"</td>
  <td width=25% style="text-align:center;">"Astronaut on the beach."</td>
  <td width=25% style="text-align:center;">"Iron man on the snow"</td>
</tr>
<!--#########################################################-->

<tr>
  <td><img src="gif_results/aaaA Stormtrooper on the sea.gif"></td>
  <td><img src="gif_results/A Iron man on the beach..gif"></td>
  <td><img src="gif_results/A astronaut on the moon.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"A Stormtrooper on the sea"</td>
  <td width=25% style="text-align:center;">"A Iron man on the beach."</td>
  <td width=25% style="text-align:center;">"A astronaut on the moon."</td>
</tr>
<!--#########################################################-->

<tr>
  <td><img src="gif_results/cdAstronaut on the beach.gif"></td>
  <td><img src="gif_results/cswSuperman on the forest.gif"></td>
  <td><img src="gif_results/cwIron man on the beach..gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"Astronaut on the beach"</td>
  <td width=25% style="text-align:center;">"Superman on the forest"</td>
  <td width=25% style="text-align:center;">"Iron man on the beach"</td>
</tr>
<!--#########################################################-->
<tr>
  <td><img src="gif_results/dfewcAstronaut on the beach.gif"></td>
  <td><img src="gif_results/ewA Robot in Antarctica.gif"></td>
  <td><img src="gif_results/vervA Stormtrooper on the sea.gif"></td>

</tr>
<tr>

</tr>
<tr>
  <td width=25% style="text-align:center;">"Astronaut on the beach"</td>
  <td width=25% style="text-align:center;">"Robot in Antarctica"</td>
  <td width=25% style="text-align:center;">"The Stormtroopers, on the beach"</td>
</tr>

<!--#########################################################-->
</table>



## 🎼🎼🎼 Citation 

```
XXXX
``` 


## 👯👯👯 Acknowledgements

This repository borrows heavily from [Tune-A-Video](https://github.com/showlab/Tune-A-Video), [FateZero](https://github.com/ChenyangQiQi/FateZero) and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/). thanks the authors for sharing their code and models.

## 🕺🕺🕺 Maintenance

This is the codebase for our research work. We are still working hard to update this repo and more details are coming in days. If you have any questions or ideas to discuss, feel free to contact [Yue Ma](y-ma21@mails.tsinghua.edu.cn) or [Yingqing He](vinthony@gmail.com) or [Xiaodong Cun](vinthony@gmail.com).

