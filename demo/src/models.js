
import Eng_GPT2_345 from './components/demos/Eng_GPT2_345';
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
            // {model: "textual-entailment", name: "Textual Entailment", component: TextualEntailment},
            // {model: "event2mind", name: "Event2Mind", component: Event2Mind},
            {model: "eng-gpt2-345", name: "GPT2 345M", component: Eng_GPT2_345, redirects: ["gpt2"]},
            // {model: "masked-lm", name: "Masked Language Modeling", component: MaskedLm}
        ]
    },
    // {
    //     label: "Contributing",
    //     iconSrc: addIcon,
    //     defaultOpen: true,
    //     models: [
    //         {model: "user-models", name: "Your model here!"}
    //     ]
    // }
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
