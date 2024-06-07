import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "Hoe werkt een stoomturbine?",
    "What is the term used for vapor that is above its boiling point?",
    "Qu'est-ce que la sublimation?"
];

const GPT4V_EXAMPLES: string[] = [
    "Hoe werkt een stoomturbine?",
    "What is the term used for vapor that is above its boiling point?",
    "Qu'est-ce que la sublimation?"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
