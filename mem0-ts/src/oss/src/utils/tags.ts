export class TagsUtils {
  static getDataPrefix = (tagNames: string[]): string => {
    if (tagNames.length > 0) {
      return `[${tagNames.join(", ")}]`;
    }
    return "";
  };

  static clarifyData = (data: string, tagNames: string[]): string => {
    return data.replace(TagsUtils.getDataPrefix(tagNames), "").trim();
  };

  static wrapData = (data: string, tagNames: string[]): string => {
    let dataPrefix = "";
    if (tagNames.length > 0) {
      dataPrefix = TagsUtils.getDataPrefix(tagNames);
    }
    if (!data.startsWith(dataPrefix)) {
      data = dataPrefix + data;
    }
    return data;
  };
}
