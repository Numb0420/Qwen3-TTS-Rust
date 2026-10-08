//! Convert explicit pronunciation markup to the pinyin text sent to the model.

const OPEN: &str = "<phoneme py=\"";
const CLOSE: &str = "</phoneme>";

pub fn prepare_text(input: &str) -> Result<String, String> {
    let mut output = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(start) = rest.find('<') {
        output.push_str(&rest[..start]);
        rest = &rest[start..];

        if rest.starts_with("<phoneme") {
            let Some(after_open) = rest.strip_prefix(OPEN) else {
                return Err("读音标记格式错误，应为 <phoneme py=\"lù\">六</phoneme>".into());
            };
            let Some(quote_end) = after_open.find("\">") else {
                return Err("读音标记缺少 py 属性的结束引号或 >".into());
            };
            let pinyin = after_open[..quote_end].trim();
            if pinyin.is_empty() || !pinyin.chars().all(valid_pinyin_char) {
                return Err("py 只能填写拼音、声调、数字和空格".into());
            }
            let after_tag = &after_open[quote_end + 2..];
            let Some(close_at) = after_tag.find(CLOSE) else {
                return Err("读音标记缺少 </phoneme>".into());
            };
            let original = &after_tag[..close_at];
            if original.trim().is_empty() || original.contains('<') || original.contains('>') {
                return Err("读音标记中的原文不能为空或包含嵌套标签".into());
            }
            output.push_str(&pinyin.split_whitespace().collect::<Vec<_>>().join(" "));
            output.push(' ');
            rest = &after_tag[close_at + CLOSE.len()..];
        } else if rest.starts_with("</phoneme") {
            return Err("发现没有起始标签的 </phoneme>".into());
        } else {
            output.push('<');
            rest = &rest[1..];
        }
    }
    output.push_str(rest);
    Ok(output)
}

fn valid_pinyin_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || c.is_whitespace()
        || c == '\''
        || matches!(c, '\u{00C0}'..='\u{024F}' | '\u{0300}'..='\u{036F}')
}

#[cfg(test)]
mod tests {
    use super::prepare_text;

    #[test]
    fn expands_multiple_annotations() {
        let text = "安徽有<phoneme py=\"bó\">亳</phoneme>州，<phoneme py=\"lù\">六</phoneme>安。";
        assert_eq!(prepare_text(text).unwrap(), "安徽有bó 州，lù 安。");
    }

    #[test]
    fn rejects_malformed_and_nested_annotations() {
        assert!(prepare_text("<phoneme py=\"lù\">六").is_err());
        assert!(prepare_text("<phoneme py=\"lù\"><phoneme py=\"ān\">安</phoneme></phoneme>").is_err());
        assert!(prepare_text("<phoneme py=\"abc\" onload=\"x\">六</phoneme>").is_err());
    }

}
