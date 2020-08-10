
import Gpt2_345 from './components/demos/Gpt2_345';
import DistilmBert from './components/demos/DistilmBert';
import BertLM from './components/demos/BertLM';
import mBert from './components/demos/mBert';
import BertFimPlus from './components/demos/BertFimPlus'
import TransformerDecoder from './components/demos/TransformerDecoder'
import TransformerDecoderSeq from './components/demos/TransformerDecoderSeq'
import annotateIcon from './icons/annotate-14px.svg';
import otherIcon from './icons/other-14px.svg';
import parseIcon from './icons/parse-14px.svg';
import passageIcon from './icons/passage-14px.svg';
import questionIcon from './icons/question-14px.svg';
import addIcon from './icons/add-14px.svg';

// This is the order in which they will appear in the menu
const modelGroups = [
    {
        label: "English LM",
        iconSrc: otherIcon,
        defaultOpen: true,
        models: [
            {model: "gpt2-345", name: "GPT2 345M", component: Gpt2_345, redirects: ["gpt2"]},
            {model: "bert-lm", name: "Bert Masked LM", component: BertLM}
        ]
    },
    {
        label: "Multilingual LM",
        iconSrc: parseIcon,
        defaultOpen: true,
        models: [
          {model: "mbert", name: "Mutilingual Bert", component: mBert},
          {model: "distilmbert", name: "Distil Mutilingual Bert", component: DistilmBert},
          {model: "bert-fimplus", name: "Mutilingual Bert finetuned for FimPlus", component: BertFimPlus}
        ]
    },
    {
      label: "Tone Prediction",
      iconSrc: passageIcon,
      defaultOpen: true,
      models: [
        // {model: "transform_decoder", name: "Autoregressive Transformer-Decoder", component: TransformerDecoder},
        {model: "transform_decoder_seq", name: "Seq2Seq Transformer-Decoder", component: TransformerDecoderSeq},
      ]
  }
]

// Create mapping from model to component
let modelComponents = {}
modelGroups.forEach((mg) => mg.models.forEach(({model, component}) => modelComponents[model] = component));

let modelRedirects = {}
modelGroups.forEach((mg) => mg.models.forEach(
  ({model, redirects}) => {
    if (redirects) {
      redirects.forEach((redirect) => modelRedirects[redirect] = model)
    }
  }
));

export { modelComponents, modelGroups, modelRedirects }
