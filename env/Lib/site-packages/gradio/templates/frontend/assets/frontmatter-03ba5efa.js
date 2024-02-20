import { s as styleTags, t as tags, f as foldNodeProp, c as foldInside, p as parseMixed, S as StreamLanguage } from './Index-f0a843d1.js';
import { yaml } from './yaml-c63fc23d.js';
import './index-7f1ea29b.js';
import './svelte/svelte.js';
import './Button-0502e555.js';
import './Index-4c82ea7b.js';
import './Check-f7edb5d9.js';
import './Copy-a69620a8.js';
import './DownloadLink-8955a15c.js';
import './file-url-a88f7e9b.js';
import './BlockLabel-d7f2598a.js';
import './Empty-df49f0b0.js';
import './Example-26f7bda8.js';

const frontMatterFence = /^---\s*$/m;
const frontmatter = {
  defineNodes: [{ name: "Frontmatter", block: true }, "FrontmatterMark"],
  props: [
    styleTags({
      Frontmatter: [tags.documentMeta, tags.monospace],
      FrontmatterMark: tags.processingInstruction
    }),
    foldNodeProp.add({
      Frontmatter: foldInside,
      FrontmatterMark: () => null
    })
  ],
  wrap: parseMixed((node) => {
    const { parser } = StreamLanguage.define(yaml);
    if (node.type.name === "Frontmatter") {
      return {
        parser,
        overlay: [{ from: node.from + 4, to: node.to - 4 }]
      };
    }
    return null;
  }),
  parseBlock: [
    {
      name: "Frontmatter",
      before: "HorizontalRule",
      parse: (cx, line) => {
        let end = void 0;
        const children = new Array();
        if (cx.lineStart === 0 && frontMatterFence.test(line.text)) {
          children.push(cx.elt("FrontmatterMark", 0, 4));
          while (cx.nextLine()) {
            if (frontMatterFence.test(line.text)) {
              end = cx.lineStart + 4;
              break;
            }
          }
          if (end !== void 0) {
            children.push(cx.elt("FrontmatterMark", end - 4, end));
            cx.addElement(cx.elt("Frontmatter", 0, end, children));
          }
          return true;
        }
        return false;
      }
    }
  ]
};

export { frontmatter };
//# sourceMappingURL=frontmatter-03ba5efa.js.map
