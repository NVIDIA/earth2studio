# E2S-`<NN>`: `<Topic>`

## Purpose

`<State the capability and the user need in one or two sentences.>`

## Contract

`<Define covered cases, exclusions, and links to related contracts.>`

### Requirements

<!-- Replace placeholders. Use a topic-specific ID prefix. Group requirements
     under descriptive subheadings only when needed. -->

| ID | Requirement |
| --- | --- |
| `<PREFIX>1` | `<Subject>` MUST `<observable behavior under a stated condition>`. |
| `<PREFIX>2` | `<Subject>` MUST reject `<invalid input>` with `<specified failure>`. |

<!-- Include public interfaces or compact tables only when needed to make the
     requirements precise. Do not prescribe internal implementation. -->

### Verification

- Given `<valid inputs>`, `<operation>` produces `<verifiable result>` (`<PREFIX>1`).
- Given `<invalid or boundary inputs>`, `<operation>` produces `<specified outcome>`
  (`<PREFIX>2`).
