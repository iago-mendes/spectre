// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions that handle parsing of input parameters.

#pragma once

#include <cerrno>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <ostream>
#include <pup.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/OptionsDetails.hpp"
#include "Options/ParseError.hpp"
#include "Options/Tags.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsMaplike.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"
#include "Utilities/TypeTraits/IsStdArrayOfSize.hpp"

namespace Options {
// Defining methods as inline in a different header from the class
// definition is somewhat strange.  It is done here to minimize the
// amount of code in the frequently-included Options.hpp file.  The
// only external consumers of Option should be create_from_yaml
// specializations, and they should only be instantiated by code in
// this file.  (Or explicitly instantiated in cpp files, which can
// include this file.)

// clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
inline Option::Option(YAML::Node node, Context context)
    : node_(std::make_unique<YAML::Node>(std::move(node))),
      context_(std::move(context)) {  // NOLINT
  context_.line = node.Mark().line;
  context_.column = node.Mark().column;
}

inline Option::Option(Context context)
    : node_(std::make_unique<YAML::Node>()), context_(std::move(context)) {}

inline const YAML::Node& Option::node() const { return *node_; }
inline const Context& Option::context() const { return context_; }

/// Append a line to the contained context.
inline void Option::append_context(const std::string& context) {
  context_.append(context);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
inline void Option::set_node(YAML::Node node) {
  // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
  *node_ = std::move(node);  // NOLINT
  context_.line = node_->Mark().line;
  context_.column = node_->Mark().column;
}

template <typename T, typename Metavariables>
T Option::parse_as() const {
  try {
    // yaml-cpp's `as` method won't parse empty nodes, so we need to
    // inline a bit of its logic.
    Options_detail::wrap_create_types<T, Metavariables> result{};
#if defined(__GNUC__) and not defined(__clang__) and __GNUC__ == 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (YAML::convert<decltype(result)>::decode(node(), result)) {
#if defined(__GNUC__) and not defined(__clang__) and __GNUC__ == 13
#pragma GCC diagnostic pop
#endif
      return Options_detail::unwrap_create_types(std::move(result));
    }
    // clang-tidy: thrown exception is not nothrow copy constructible
    throw YAML::BadConversion(node().Mark());  // NOLINT
  } catch (const YAML::BadConversion& e) {
    // This happens when trying to parse an empty value as a container
    // with no entries.
    if ((tt::is_a_v<std::vector, T> or tt::is_std_array_of_size_v<0, T> or
         tt::is_maplike_v<T>) and node().IsNull()) {
      return T{};
    }
    Context error_context = context();
    error_context.line = e.mark.line;
    error_context.column = e.mark.column;
    std::ostringstream ss;
    ss << "Failed to convert value to type "
       << Options_detail::yaml_type<T>::value() << ":";

    const std::string value_text = YAML::Dump(node());
    if (value_text.find('\n') == std::string::npos) {
      ss << " " << value_text;
    } else {
      // Indent each line of the value by two spaces and start on a new line
      ss << "\n  ";
      for (char c : value_text) {
        ss << c;
        if (c == '\n') {
          ss << "  ";
        }
      }
    }

    if (tt::is_a_v<std::vector, T> or tt::is_std_array_v<T>) {
      ss << "\n\nNote: For sequences this can happen because the length of the "
            "sequence specified\nin the input file is not equal to the length "
            "expected by the code. Sequences in\nfiles can be denoted either "
            "as a bracket enclosed list ([foo, bar]) or with each\nentry on a "
            "separate line, indented and preceeded by a dash (  - foo).";
    }
    PARSE_ERROR(error_context, ss.str());
  } catch (const Options::detail::propagate_context& e) {
    Context error_context = context();
    // Avoid line numbers in the middle of the trace
    error_context.line = -1;
    error_context.column = -1;
    PARSE_ERROR(error_context, e.message());
  } catch (std::exception& e) {
    ERROR("Unexpected exception: " << e.what());
  }
}

namespace Options_detail {
template <typename T, typename Metavariables, typename Subgroup>
struct get_impl;
}  // namespace Options_detail

/// \ingroup OptionParsingGroup
/// \brief Class that handles parsing an input file
///
/// Options must be given YAML data to parse before output can be
/// extracted.  This can be done either from a file (parse_file
/// method), from a string (parse method), or, in the case of
/// recursive parsing, from an Option (parse method).  The options
/// can then be extracted using the get method.
///
/// \example
/// \snippet Test_Options.cpp options_example_scalar_struct
/// \snippet Test_Options.cpp options_example_scalar_parse
///
/// \see the \ref dev_guide_option_parsing tutorial
///
/// \tparam OptionList the list of option structs to parse
/// \tparam Group the option group with a group hierarchy
template <typename OptionList, typename Group = NoSuchType>
class Parser {
 private:
  /// All top-level options and top-level groups of options. Every option in
  /// `OptionList` is either in this list or in the hierarchy of one of the
  /// groups in this list.
  using tags_and_subgroups_list = tmpl::remove_duplicates<tmpl::transform<
      OptionList, Options_detail::find_subgroup<tmpl::_1, tmpl::pin<Group>>>>;

 public:
  Parser() = default;

  /// \param help_text an overall description of the options
  explicit Parser(std::string help_text);

  /// Parse a string to obtain options and their values.
  ///
  /// \param options the string holding the YAML formatted options
  void parse(std::string options);

  /// Parse an Option to obtain options and their values.
  void parse(const Option& options);

  /// Parse a file containing options
  ///
  /// \param file_name the path to the file to parse
  /// \param require_metadata require the input file to have a metadata section
  void parse_file(const std::string& file_name, bool require_metadata = true);

  /// Overlay the options from a string or file on top of the
  /// currently parsed options.
  ///
  /// Any tag included in the list passed as the template parameter
  /// can be overridden by a new parsed value.  Newly parsed options
  /// replace the previous values.  Any tags not appearing in the new
  /// input are left unchanged.
  /// @{
  template <typename OverlayOptions>
  void overlay(std::string options);

  template <typename OverlayOptions>
  void overlay_file(const std::string& file_name);
  /// @}

  /// Get the value of the specified option
  ///
  /// \tparam T the option to retrieve
  /// \return the value of the option
  template <typename T, typename Metavariables = NoSuchType>
  typename T::type get() const;

  /// Call a function with the specified options as arguments.
  ///
  /// \tparam TagList a typelist of options to pass
  /// \return the result of the function call
  template <typename TagList, typename Metavariables = NoSuchType, typename F>
  decltype(auto) apply(F&& func) const;

  /// Call a function with the typelist of parsed options (i.e., the
  /// supplied option list with the chosen branches of any
  /// Alternatives inlined) and the option values as arguments.
  ///
  /// \return the result of the function call.  This must have the
  /// same type for all valid sets of parsed arguments.
  template <typename Metavariables = NoSuchType, typename F>
  decltype(auto) apply_all(F&& func) const;

  /// Get the help string
  template <typename TagsAndSubgroups = tags_and_subgroups_list>
  std::string help() const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <typename, typename>
  friend class Parser;
  template <typename, typename, typename>
  friend struct Options_detail::get_impl;

  static_assert(tt::is_a<tmpl::list, OptionList>::value,
                "The OptionList template parameter to Options must be a "
                "tmpl::list<...>.");

  // All options that could be specified, including those that have
  // alternatives and are therefore not required.
  using all_possible_options = tmpl::remove_duplicates<
      typename Options_detail::flatten_alternatives<OptionList>::type>;

  static_assert(
      std::is_same_v<
          typename Options_detail::flatten_alternatives<OptionList>::type,
          OptionList> or
          tmpl::all<
              all_possible_options,
              std::is_same<tmpl::_1, Options_detail::find_subgroup<
                                         tmpl::_1, tmpl::pin<Group>>>>::value,
      "Option parser cannot handle Alternatives and options with groups "
      "simultaneously.");

  /// All top-level subgroups
  using subgroups = tmpl::list_difference<tags_and_subgroups_list, OptionList>;

  // The maximum length of an option label.
  static constexpr int max_label_size_ = 70;

  /// Parse a YAML node containing options
  void parse(const YAML::Node& node);

  /// Overlay data from a YAML node
  template <typename OverlayOptions>
  void overlay(const YAML::Node& node);

  /// Check that the size is not smaller than the lower bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_lower_bound_on_size(const typename T::type& t,
                                 const Context& context) const;

  /// Check that the size is not larger than the upper bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_upper_bound_on_size(const typename T::type& t,
                                 const Context& context) const;

  /// If the options has a lower bound, check it is satisfied.
  ///
  /// Note: Lower bounds are >=, not just >.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_lower_bound(const typename T::type& t,
                         const Context& context) const;

  /// If the options has a upper bound, check it is satisfied.
  ///
  /// Note: Upper bounds are <=, not just <.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_upper_bound(const typename T::type& t,
                         const Context& context) const;

  /// Get the help string for parsing errors
  template <typename TagsAndSubgroups = tags_and_subgroups_list>
  std::string parsing_help(const YAML::Node& options) const;

  /// Error message when failed to parse an input file.
  [[noreturn]] void parser_error(const YAML::Exception& e) const;

  template <typename ChosenOptions, typename RemainingOptions, typename F>
  auto call_with_chosen_alternatives_impl(F&& func,
                                          std::vector<size_t> choices) const;

  template <typename F>
  auto call_with_chosen_alternatives(F&& func) const {
    return call_with_chosen_alternatives_impl<tmpl::list<>, OptionList>(
        std::forward<F>(func), alternative_choices_);
  }

  std::string help_text_{};
  Context context_{};
  std::vector<std::string> input_source_{};
  std::unordered_map<std::string, YAML::Node> parsed_options_{};

  template <typename Subgroup>
  struct SubgroupParser {
    using type = Parser<Options_detail::options_in_group<OptionList, Subgroup>,
                        Subgroup>;
  };

  tuples::tagged_tuple_from_typelist<
      tmpl::transform<subgroups, tmpl::bind<SubgroupParser, tmpl::_1>>>
      subgroup_parsers_ =
          tmpl::as_pack<subgroups>([this](auto... subgroup_tags) {
            (void)this;  // gcc wants this for subgroup_parsers_
            return decltype(subgroup_parsers_)(
                tmpl::type_from<decltype(subgroup_tags)>::help...);
          });

  // The choices made for option alternatives in a depth-first order.
  // Starting from the front of the option list, when reaching the
  // first Alternatives object, replace it with the options in the nth
  // choice, where n is the *last* element of this vector.  Continue
  // processing from the start of those options, using the second to
  // last value here for the next choice, and so on.
  std::vector<size_t> alternative_choices_{};
};

template <typename OptionList, typename Group>
Parser<OptionList, Group>::Parser(std::string help_text)
    : help_text_(std::move(help_text)) {
  tmpl::for_each<all_possible_options>([](auto t) {
    using T = typename decltype(t)::type;
    const std::string label = pretty_type::name<T>();
    ASSERT(label.size() <= max_label_size_,
           "The option name " << label
                              << " is too long for nice formatting, "
                                 "please shorten the name to "
                              << max_label_size_ << " characters or fewer");
    ASSERT(std::strlen(T::help) > 0,
           "You must supply a help string of non-zero length for " << label);
  });
}

namespace detail {
YAML::Node load_and_check_yaml(const std::string& options,
                               bool require_metadata);
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(std::string options) {
  context_.append("In string");
  input_source_.push_back(std::move(options));
  try {
    parse(detail::load_and_check_yaml(input_source_.back(), false));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(const Option& options) {
  context_ = options.context();
  parse(options.node());
}

namespace Options_detail {
inline std::ifstream open_file(const std::string& file_name) {
  errno = 0;
  std::ifstream input(file_name);
  if (not input) {
    // There is no standard way to get an error message from an
    // fstream, but this works on many implementations.
    ERROR("Could not open " << file_name << ": " << strerror(errno));
  }
  return input;
}
}  // namespace Options_detail

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse_file(const std::string& file_name,
                                           const bool require_metadata) {
  context_.append("In " + file_name);
  auto input = Options_detail::open_file(file_name);
  input_source_.push_back(std::string(std::istreambuf_iterator(input), {}));
  try {
    parse(detail::load_and_check_yaml(input_source_.back(), require_metadata));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
template <typename OverlayOptions>
void Parser<OptionList, Group>::overlay(std::string options) {
  context_ = Context{};
  context_.append("In string");
  input_source_.push_back(std::move(options));
  try {
    overlay<OverlayOptions>(
        detail::load_and_check_yaml(input_source_.back(), false));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
template <typename OverlayOptions>
void Parser<OptionList, Group>::overlay_file(const std::string& file_name) {
  context_ = Context{};
  context_.append("In " + file_name);
  auto input = Options_detail::open_file(file_name);
  input_source_.push_back(std::string(std::istreambuf_iterator(input), {}));
  try {
    overlay<OverlayOptions>(
        detail::load_and_check_yaml(input_source_.back(), false));
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

namespace Options_detail {
// Attempts to match the given_options against OptionList, choosing
// between any alternatives to maximize the number of matches.
// Returns the highest number of matches and the choices required to
// obtain that number.  (See the description of
// Parser::alternative_choices_ for the format of the choices.)
//
// In the case of multiple equally good matches, the choice between
// them will be given as std::numeric_limits<size_t>::max().  This may
// cause a failure later or may be ignored if a better choice is
// found.
//
// This does not handle groups correctly, but we disallow alternatives
// when groups are present so there is only one possible choice that
// this function could make.
template <typename OptionList>
std::pair<int, std::vector<size_t>> choose_alternatives(
    const std::unordered_set<std::string>& given_options) {
  int num_matched = 0;
  std::vector<size_t> alternative_choices{};
  tmpl::for_each<OptionList>([&alternative_choices, &num_matched,
                              &given_options](auto opt) {
    using Opt = tmpl::type_from<decltype(opt)>;
    if constexpr (not tt::is_a_v<Options::Alternatives, Opt>) {
      if (given_options.count(pretty_type::name<Opt>()) == 1) {
        ++num_matched;
      }
    } else {
      int most_matches = 0;
      std::vector<size_t> best_alternatives{std::numeric_limits<size_t>::max()};

      size_t alternative_number = 0;
      tmpl::for_each<Opt>([&alternative_number, &best_alternatives,
                           &most_matches, &given_options](auto alternative) {
        using Alternative = tmpl::type_from<decltype(alternative)>;
        auto alternative_match =
            choose_alternatives<Alternative>(given_options);
        if (alternative_match.first > most_matches) {
          most_matches = alternative_match.first;
          alternative_match.second.push_back(alternative_number);
          best_alternatives = std::move(alternative_match.second);
        } else if (alternative_match.first == most_matches) {
          // Two equally good matches
          best_alternatives.clear();
          best_alternatives.push_back(std::numeric_limits<size_t>::max());
        }
        ++alternative_number;
      });
      num_matched += most_matches;
      alternative_choices.insert(alternative_choices.begin(),
                                 best_alternatives.begin(),
                                 best_alternatives.end());
    }
  });
  return {num_matched, std::move(alternative_choices)};
}
}  // namespace Options_detail

namespace Options_detail {
template <typename Tag, typename Metavariables, typename Subgroup>
struct get_impl {
  template <typename OptionList, typename Group>
  static typename Tag::type apply(const Parser<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<OptionList, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    return tuples::get<typename Parser<
        OptionList, Group>::template SubgroupParser<Subgroup>>(
               opts.subgroup_parsers_)
        .template get<Tag, Metavariables>();
  }
};

template <typename Tag, typename Metavariables>
struct get_impl<Tag, Metavariables, Tag> {
  template <typename OptionList, typename Group>
  static typename Tag::type apply(const Parser<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<
            typename Parser<OptionList, Group>::all_possible_options, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    const std::string label = pretty_type::name<Tag>();

    const auto supplied_option = opts.parsed_options_.find(label);
    ASSERT(supplied_option != opts.parsed_options_.end(),
           "Requested option from alternative that was not supplied.");
    Option option(supplied_option->second, opts.context_);
    option.append_context("While parsing option " + label);

    auto t = option.parse_as<typename Tag::type, Metavariables>();

    if constexpr (Options_detail::has_suggested<Tag>::value) {
      static_assert(
          std::is_same_v<decltype(Tag::suggested_value()), typename Tag::type>,
          "Suggested value is not of the same type as the option.");

      // This can be easily relaxed, but using it would require
      // writing comparison operators for abstract base classes.  If
      // someone wants this enough to go though the effort of doing
      // that, it would just require comparing the dereferenced
      // pointers below to decide whether the suggestion was followed.
      static_assert(not tt::is_a_v<std::unique_ptr, typename Tag::type>,
                    "Suggestions are not supported for pointer types.");

      const auto suggested_value = Tag::suggested_value();
      {
        Context context;
        context.append("Checking SUGGESTED value for " +
                       pretty_type::name<Tag>());
        opts.template check_lower_bound_on_size<Tag>(suggested_value, context);
        opts.template check_upper_bound_on_size<Tag>(suggested_value, context);
        opts.template check_lower_bound<Tag>(suggested_value, context);
        opts.template check_upper_bound<Tag>(suggested_value, context);
      }

      if (t != suggested_value) {
        Parallel::printf_error(
            "%s, line %d:\n  Specified: %s\n  Suggested: %s\n",
            label, option.context().line + 1,
            (MakeString{} << std::boolalpha << t),
            (MakeString{} << std::boolalpha << suggested_value));
      }
    }

    opts.template check_lower_bound_on_size<Tag>(t, option.context());
    opts.template check_upper_bound_on_size<Tag>(t, option.context());
    opts.template check_lower_bound<Tag>(t, option.context());
    opts.template check_upper_bound<Tag>(t, option.context());
    return t;
  }
};

template <typename Metavariables>
struct get_impl<Tags::InputSource, Metavariables, Tags::InputSource> {
  template <typename OptionList, typename Group>
  static Tags::InputSource::type apply(const Parser<OptionList, Group>& opts) {
    return opts.input_source_;
  }
};
}  // namespace Options_detail

template <typename OptionList, typename Group>
template <typename Tag, typename Metavariables>
typename Tag::type Parser<OptionList, Group>::get() const {
  return Options_detail::get_impl<
      Tag, Metavariables,
      typename Options_detail::find_subgroup<Tag, Group>::type>::apply(*this);
}

namespace Options_detail {
template <typename>
struct apply_helper;

template <typename... Tags>
struct apply_helper<tmpl::list<Tags...>> {
  template <typename Metavariables, typename Options, typename F>
  static decltype(auto) apply(const Options& opts, F&& func) {
    return func(opts.template get<Tags, Metavariables>()...);
  }
};
}  // namespace Options_detail

/// \cond
// Doxygen is confused by decltype(auto)
template <typename OptionList, typename Group>
template <typename TagList, typename Metavariables, typename F>
decltype(auto) Parser<OptionList, Group>::apply(F&& func) const {
  return Options_detail::apply_helper<TagList>::template apply<Metavariables>(
      *this, std::forward<F>(func));
}

template <typename OptionList, typename Group>
template <typename Metavariables, typename F>
decltype(auto) Parser<OptionList, Group>::apply_all(F&& func) const {
  return call_with_chosen_alternatives([this, &func](
                                           auto chosen_alternatives /*meta*/) {
    using ChosenAlternatives = decltype(chosen_alternatives);
    return this->apply<ChosenAlternatives, Metavariables>([&func](
                                                              auto&&... args) {
      return std::forward<F>(func)(ChosenAlternatives{}, std::move(args)...);
    });
  });
}
/// \endcond

template <typename OptionList, typename Group>
template <typename TagsAndSubgroups>
std::string Parser<OptionList, Group>::help() const {
  std::ostringstream ss;
  ss << "\n==== Description of expected options:\n" << help_text_;
  if (tmpl::size<TagsAndSubgroups>::value > 0) {
    ss << "\n\nOptions:\n"
       << Options_detail::print<OptionList, TagsAndSubgroups>::apply("  ");
  } else {
    ss << "\n\n<No options>\n";
  }
  return ss.str();
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::pup(PUP::er& p) {
  static_assert(std::is_same_v<Group, NoSuchType>,
                "Inner parsers should be recreated by the root parser, not "
                "serialized.");
  // We reconstruct most of the state when deserializing, rather than
  // trying to package it.
  p | help_text_;
  p | input_source_;
  if (p.isUnpacking() and not input_source_.empty()) {
    // input_source_ is populated by the `parse` and `overlay` calls
    // below, so we have to clear out the old values before calling
    // them.
    auto received_source = std::move(input_source_);
    input_source_.clear();
    parse(std::move(received_source[0]));
    for (size_t i = 1; i < received_source.size(); ++i) {
      overlay<OptionList>(std::move(received_source[i]));
    }
  }
}

// implementation of Options::Parser::parse() factored out to save on compile
// time and compile memory
namespace parse_detail {
std::unordered_set<std::string> get_given_options(
    const Options::Context& context, const YAML::Node& node,
    const std::string& help);

void check_for_unique_choice(const std::vector<size_t>& alternative_choices,
                             const Options::Context& context,
                             const std::string& parsing_help);

void add_name_to_valid_option_names(
    gsl::not_null<std::vector<std::string>*> valid_option_names,
    const std::string& label);

template <typename TopLevelOptionsAndGroups>
struct get_valid_option_names;

template <typename... TopLevelOptionsAndGroups>
struct get_valid_option_names<tmpl::list<TopLevelOptionsAndGroups...>> {
  static std::vector<std::string> apply() {
    // Use an ordered container so the missing options are reported in
    // the order they are given in the help string.
    std::vector<std::string> valid_option_names;
    valid_option_names.reserve(
        tmpl::size<tmpl::list<TopLevelOptionsAndGroups...>>{});
    (add_name_to_valid_option_names(
         make_not_null(&valid_option_names),
         pretty_type::name<TopLevelOptionsAndGroups>()),
     ...);
    return valid_option_names;
  }
};

[[noreturn]] void option_specified_twice_error(const Options::Context& context,
                                               const std::string& name,
                                               const std::string& parsing_help);

[[noreturn]] void unused_key_error(const Context& context,
                                   const std::string& name,
                                   const std::string& parsing_help);

template <typename Tag>
void check_for_unused_key(const Context& context, const std::string& name,
                          const std::string& parsing_help) {
  if (name == pretty_type::name<Tag>()) {
    unused_key_error(context, name, parsing_help);
  }
}

template <typename AllPossibleOptions>
struct check_for_unused_key_helper;

template <typename... AllPossibleOptions>
struct check_for_unused_key_helper<tmpl::list<AllPossibleOptions...>> {
  static void apply(const Context& context, const std::string& name,
                    const std::string& parsing_help) {
    (check_for_unused_key<AllPossibleOptions>(context, name, parsing_help),
     ...);
  }
};

[[noreturn]] void option_invalid_error(const Options::Context& context,
                                       const std::string& name,
                                       const std::string& parsing_help);

void check_for_missing_option(const std::vector<std::string>& valid_names,
                              const Options::Context& context,
                              const std::string& parsing_help);

std::string add_group_prefix_to_name(const std::string& name);

void print_top_level_error_message();
}  // namespace parse_detail

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(const YAML::Node& node) {
  std::unordered_set<std::string> given_options =
      parse_detail::get_given_options(context_, node, help());

  alternative_choices_ =
      Options_detail::choose_alternatives<OptionList>(given_options).second;
  const std::string parsing_help_message = parsing_help(node);
  parse_detail::check_for_unique_choice(alternative_choices_, context_,
                                        parsing_help_message);

  auto valid_names = call_with_chosen_alternatives([](auto option_list_v) {
    using option_list = decltype(option_list_v);
    using top_level_options_and_groups =
        tmpl::remove_duplicates<tmpl::transform<
            option_list,
            Options_detail::find_subgroup<tmpl::_1, tmpl::pin<Group>>>>;
    return parse_detail::get_valid_option_names<
        top_level_options_and_groups>::apply();
  });

  for (const auto& name_and_value : node) {
    const auto& name = name_and_value.first.as<std::string>();
    const auto& value = name_and_value.second;
    auto context = context_;
    context.line = name_and_value.first.Mark().line;
    context.column = name_and_value.first.Mark().column;

    // Check for duplicate key
    if (0 != parsed_options_.count(name)) {
      parse_detail::option_specified_twice_error(context, name,
                                                 parsing_help_message);
    }

    // Check for invalid key
    const auto name_it = alg::find(valid_names, name);
    if (name_it == valid_names.end()) {
      parse_detail::check_for_unused_key_helper<all_possible_options>::apply(
          context, name, parsing_help_message);
      parse_detail::option_invalid_error(context, name, parsing_help_message);
    }

    parsed_options_.emplace(name, value);
    valid_names.erase(name_it);
  }

  parse_detail::check_for_missing_option(valid_names, context_,
                                         parsing_help_message);

  tmpl::for_each<subgroups>([this](auto subgroup_v) {
    using subgroup = tmpl::type_from<decltype(subgroup_v)>;
    auto& subgroup_parser =
        tuples::get<SubgroupParser<subgroup>>(subgroup_parsers_);
    subgroup_parser.context_ = context_;
    subgroup_parser.context_.append(
        parse_detail::add_group_prefix_to_name(pretty_type::name<subgroup>()));
    subgroup_parser.parse(
        parsed_options_.find(pretty_type::name<subgroup>())->second);
  });

  // Any actual warnings will be printed by later calls to get or
  // apply, but it is not clear how to determine in those functions
  // whether this message should be printed.
  if constexpr (std::is_same_v<Group, NoSuchType>) {
    if (context_.top_level) {
      parse_detail::print_top_level_error_message();
    }
  }
}

template <typename OptionList, typename Group>
template <typename OverlayOptions>
void Parser<OptionList, Group>::overlay(const YAML::Node& node) {
  // This could be relaxed to allow mandatory options in a list with
  // alternatives to be overlaid (or even any options in the chosen
  // alternative), but overlaying is only done at top level and we
  // don't use alternatives there (because they conflict with groups,
  // which we do use).
  static_assert(
      std::is_same_v<
          typename Options_detail::flatten_alternatives<OptionList>::type,
          OptionList>,
      "Cannot overlay options when using alternatives.");
  static_assert(
      std::is_same_v<tmpl::list_difference<OverlayOptions, OptionList>,
                     tmpl::list<>>,
      "Can only overlay options that were originally parsed.");

  using overlayable_tags_and_subgroups_list =
      tmpl::remove_duplicates<tmpl::transform<
          OverlayOptions,
          Options_detail::find_subgroup<tmpl::_1, tmpl::pin<Group>>>>;

  if (not(node.IsMap() or node.IsNull())) {
    PARSE_ERROR(context_, "'" << node << "' does not look like options.\n"
                              << help<overlayable_tags_and_subgroups_list>());
  }

  std::unordered_set<std::string> overlaid_options{};
  overlaid_options.reserve(node.size());

  for (const auto& name_and_value : node) {
    const auto& name = name_and_value.first.as<std::string>();
    const auto& value = name_and_value.second;
    auto context = context_;
    context.line = name_and_value.first.Mark().line;
    context.column = name_and_value.first.Mark().column;

    if (tmpl::as_pack<tags_and_subgroups_list>([&name](auto... opts) {
          return (
              (name != pretty_type::name<tmpl::type_from<decltype(opts)>>()) and
              ...);
        })) {
      PARSE_ERROR(context,
                  "Option '" << name << "' is not a valid option.\n"
                  << parsing_help<overlayable_tags_and_subgroups_list>(node));
    }

    if (tmpl::as_pack<overlayable_tags_and_subgroups_list>(
            [&name](auto... opts) {
              return ((name !=
                       pretty_type::name<tmpl::type_from<decltype(opts)>>()) and
                      ...);
            })) {
      PARSE_ERROR(context,
                  "Option '" << name << "' is not overlayable.\n"
                  << parsing_help<overlayable_tags_and_subgroups_list>(node));
    }

    // Check for duplicate key
    if (0 != overlaid_options.count(name)) {
      PARSE_ERROR(context,
                  "Option '" << name << "' specified twice.\n"
                  << parsing_help<overlayable_tags_and_subgroups_list>(node));
    }

    overlaid_options.insert(name);
    parsed_options_.at(name) = value;
  }

  tmpl::for_each<subgroups>([this, &overlaid_options](auto subgroup_v) {
    using subgroup = tmpl::type_from<decltype(subgroup_v)>;
    if (overlaid_options.count(pretty_type::name<subgroup>()) == 1) {
      auto& subgroup_parser =
          tuples::get<SubgroupParser<subgroup>>(subgroup_parsers_);
      subgroup_parser.template overlay<
          Options_detail::options_in_group<OverlayOptions, subgroup>>(
          parsed_options_.find(pretty_type::name<subgroup>())->second);
    }
  });
}

template <typename OptionList, typename Group>
template <typename T>
void Parser<OptionList, Group>::check_lower_bound_on_size(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_lower_bound_on_size<T>::value) {
    static_assert(std::is_same_v<decltype(T::lower_bound_on_size()), size_t>,
                  "lower_bound_on_size() is not a size_t.");
    if (t.size() < T::lower_bound_on_size()) {
      PARSE_ERROR(context, "Value must have at least "
                               << T::lower_bound_on_size() << " entries, but "
                               << t.size() << " were given.\n"
                               << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
void Parser<OptionList, Group>::check_upper_bound_on_size(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_upper_bound_on_size<T>::value) {
    static_assert(std::is_same_v<decltype(T::upper_bound_on_size()), size_t>,
                  "upper_bound_on_size() is not a size_t.");
    if (t.size() > T::upper_bound_on_size()) {
      PARSE_ERROR(context, "Value must have at most "
                               << T::upper_bound_on_size() << " entries, but "
                               << t.size() << " were given.\n"
                               << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
inline void Parser<OptionList, Group>::check_lower_bound(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_lower_bound<T>::value) {
    static_assert(std::is_same_v<decltype(T::lower_bound()), typename T::type>,
                  "Lower bound is not of the same type as the option.");
    static_assert(not std::is_same_v<typename T::type, bool>,
                  "Cannot set a lower bound for a bool.");
    if (t < T::lower_bound()) {
      PARSE_ERROR(context, "Value " << (MakeString{} << t)
                                    << " is below the lower bound of "
                                    << (MakeString{} << T::lower_bound())
                                    << ".\n" << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
inline void Parser<OptionList, Group>::check_upper_bound(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_upper_bound<T>::value) {
    static_assert(std::is_same_v<decltype(T::upper_bound()), typename T::type>,
                  "Upper bound is not of the same type as the option.");
    static_assert(not std::is_same_v<typename T::type, bool>,
                  "Cannot set an upper bound for a bool.");
    if (t > T::upper_bound()) {
      PARSE_ERROR(context, "Value " << (MakeString{} << t)
                                    << " is above the upper bound of "
                                    << (MakeString{} << T::upper_bound())
                                    << ".\n" << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename TagsAndSubgroups>
std::string Parser<OptionList, Group>::parsing_help(
    const YAML::Node& options) const {
  std::ostringstream os;
  // At top level this would dump the entire input file, which is very
  // verbose and not very informative.  At lower levels the result
  // should be much shorter and may actually give useful context for
  // what part of the file is being parsed.
  if (not context_.top_level) {
    os << "\n==== Parsing the option string:\n" << options << "\n";
  }
  os << help<TagsAndSubgroups>();
  return os.str();
}

template <typename OptionList, typename Group>
[[noreturn]] void Parser<OptionList, Group>::parser_error(
    const YAML::Exception& e) const {
  auto context = context_;
  context.line = e.mark.line;
  context.column = e.mark.column;
  // Inline the top_level branch of PARSE_ERROR to avoid warning that
  // the other branch would call terminate.  (Parser errors can only
  // be generated at top level.)
  ERROR(
      "\n"
      << context
      << "Unable to correctly parse the input file because of a syntax error.\n"
         "This is often due to placing a suboption on the same line as an "
         "option, e.g.:\nDomainCreator: CreateInterval:\n  IsPeriodicIn: "
         "[false]\n\nShould be:\nDomainCreator:\n  CreateInterval:\n    "
         "IsPeriodicIn: [true]\n\nSee an example input file for help.");
}

template <typename OptionList, typename Group>
template <typename ChosenOptions, typename RemainingOptions, typename F>
auto Parser<OptionList, Group>::call_with_chosen_alternatives_impl(
    F&& func, std::vector<size_t> choices) const {
  if constexpr (std::is_same_v<RemainingOptions, tmpl::list<>>) {
    return std::forward<F>(func)(ChosenOptions{});
  } else {
    using next_option = tmpl::front<RemainingOptions>;
    using remaining_options = tmpl::pop_front<RemainingOptions>;

    if constexpr (not tt::is_a_v<Options::Alternatives, next_option>) {
      return call_with_chosen_alternatives_impl<
          tmpl::push_back<ChosenOptions, next_option>, remaining_options>(
          std::forward<F>(func), std::move(choices));
    } else {
      using Result =
          decltype(call_with_chosen_alternatives_impl<
                   ChosenOptions,
                   tmpl::append<tmpl::front<next_option>, remaining_options>>(
              std::forward<F>(func), std::move(choices)));

      const size_t choice = choices.back();
      choices.pop_back();

      Result result{};
      size_t alternative_number = 0;
      tmpl::for_each<next_option>([this, &alternative_number, &choice, &choices,
                                   &func, &result](auto alternative) {
        using Alternative = tmpl::type_from<decltype(alternative)>;
        if (choice == alternative_number++) {
          result = this->call_with_chosen_alternatives_impl<
              ChosenOptions, tmpl::append<Alternative, remaining_options>>(
              std::forward<F>(func), std::move(choices));
        }
      });
      return result;
    }
  }
}

namespace Options_detail {
// Work around Clang bug: https://github.com/llvm/llvm-project/issues/33002
template <typename... T>
struct my_void_t_impl {
  using type = void;
};
template <typename... T>
using my_void_t = typename my_void_t_impl<T...>::type;

template <typename T, typename Metavariables, typename = my_void_t<>>
struct has_options_list : std::false_type {};

template <typename T, typename Metavariables>
struct has_options_list<T, Metavariables,
                        my_void_t<typename T::template options<Metavariables>>>
    : std::true_type {};

template <typename T, typename Metavariables>
struct has_options_list<T, Metavariables, my_void_t<typename T::options>>
    : std::true_type {};

template <typename T, typename Metavariables, typename = std::void_t<>>
struct get_options_list {
  using type = typename T::template options<Metavariables>;
};

template <typename T, typename Metavariables>
struct get_options_list<T, Metavariables, std::void_t<typename T::options>> {
  using type = typename T::options;
};

template <typename T, typename Metavariables>
struct ClassConstructor {
  const Options::Context& context;

  template <typename ParsedOptions, typename... Args>
  T operator()(ParsedOptions /*meta*/, Args&&... args) const {
    if constexpr (std::is_constructible<T, ParsedOptions,
                                        decltype(std::move(args))...,
                                        const Context&, Metavariables>{}) {
      return T(ParsedOptions{}, std::move(args)..., context, Metavariables{});
    } else if constexpr (std::is_constructible<T, ParsedOptions,
                                               decltype(std::move(args))...,
                                               const Context&>{}) {
      return T(ParsedOptions{}, std::move(args)..., context);
    } else if constexpr (std::is_constructible<T, ParsedOptions,
                                               decltype(std::move(
                                                   args))...>{}) {
      return T(ParsedOptions{}, std::move(args)...);
    } else if constexpr (std::is_constructible<T, decltype(std::move(args))...,
                                               const Context&,
                                               Metavariables>{}) {
      return T(std::move(args)..., context, Metavariables{});
    } else if constexpr (std::is_constructible<T, decltype(std::move(args))...,
                                               const Context&>{}) {
      return T(std::move(args)..., context);
    } else {
      return T{std::move(args)...};
    }
  }
};
}  // namespace Options_detail

template <typename T>
template <typename Metavariables>
T create_from_yaml<T>::create(const Option& options) {
  Parser<typename Options_detail::get_options_list<T, Metavariables>::type>
      parser(T::help);
  parser.parse(options);
  return parser.template apply_all<Metavariables>(
      Options_detail::ClassConstructor<T, Metavariables>{options.context()});
}

// yaml-cpp doesn't handle C++11 types yet
template <typename K, typename V, typename H, typename P>
struct create_from_yaml<std::unordered_map<K, V, H, P>> {
  template <typename Metavariables>
  static std::unordered_map<K, V, H, P> create(const Option& options) {
    auto ordered = options.parse_as<std::map<K, V>, Metavariables>();
    std::unordered_map<K, V, H, P> result;
    for (auto it = ordered.begin(); it != ordered.end();) {
      auto node = ordered.extract(it++);
      result.emplace(std::move(node.key()), std::move(node.mapped()));
    }
    return result;
  }
};

namespace Options_detail {
// To get the full parse backtrace for a variant parse error the
// failure should occur inside a nested call to parse_as.  This is a
// type that will produce the correct error by failing to parse.
template <typename... T>
struct variant_parse_error {};

template <typename... T>
struct yaml_type<variant_parse_error<T...>> : yaml_type<std::variant<T...>> {};
}  // namespace Options_detail

template <typename... T>
struct create_from_yaml<Options::Options_detail::variant_parse_error<T...>> {
  template <typename Metavariables>
  [[noreturn]] static Options::Options_detail::variant_parse_error<T...> create(
      const Option& options) {
    throw YAML::BadConversion(options.node().Mark());
  }
};

namespace Options_variant_detail {
template <typename T, typename Metavariables,
          typename =
              typename Options_detail::has_options_list<T, Metavariables>::type>
struct is_alternative_parsable_impl : std::false_type {};

template <typename T, typename Metavariables>
struct is_alternative_parsable_impl<T, Metavariables, std::true_type>
    : tmpl::all<
          typename Options_detail::get_options_list<T, Metavariables>::type,
          std::is_same<tmpl::_1,
                       Options_detail::find_subgroup<tmpl::_1, void>>> {};

template <typename T, typename Metavariables>
struct is_alternative_parsable
    : is_alternative_parsable_impl<T, Metavariables> {};

template <typename Result, typename Metavariables, typename... Alternatives>
Result parse_as_alternatives(const Options::Option& options,
                             tmpl::list<Alternatives...> /*meta*/) {
  using options_list = tmpl::list<
      Options::Alternatives<typename Options_detail::get_options_list<
          Alternatives, Metavariables>::type...>>;
  std::string help = ("" + ... +
                      (Options_detail::yaml_type<Alternatives>::value() + "\n" +
                       wrap_text(Alternatives::help, 77, "  ") + "\n\n"));
  help.resize(help.size() - 2);
  Options::Parser<options_list> parser(std::move(help));
  parser.parse(options);
  return parser.template apply_all<Metavariables>([&](auto parsed_options,
                                                      auto... args) -> Result {
    // Actually matching against the whole option list is hard in the
    // presence of possible nested alternatives, so we just check if
    // all the options are individually valid.
    using possible_alternatives = tmpl::filter<
        tmpl::list<Alternatives...>,
        std::is_same<tmpl::bind<tmpl::list_difference,
                                tmpl::pin<decltype(parsed_options)>,
                                Options_detail::flatten_alternatives<
                                    Options_detail::get_options_list<
                                        tmpl::_1, tmpl::pin<Metavariables>>>>,
                     tmpl::pin<tmpl::list<>>>>;
    static_assert(tmpl::size<possible_alternatives>::value == 1,
                  "Option lists for variant alternatives are too similar.");
    return Options_detail::ClassConstructor<tmpl::front<possible_alternatives>,
                                            Metavariables>{options.context()}(
        parsed_options, std::move(args)...);
  });
}
}  // namespace Options_variant_detail

template <typename... T>
struct create_from_yaml<std::variant<T...>> {
  using Result = std::variant<T...>;
  static_assert(std::is_same_v<tmpl::list<T...>,
                               tmpl::remove_duplicates<tmpl::list<T...>>>,
                "Cannot parse variants with duplicate types.");

  template <typename Metavariables>
  static Result create(const Option& options) {
    using alternative_parsable_types =
        tmpl::filter<tmpl::list<T...>,
                     Options_variant_detail::is_alternative_parsable<
                         tmpl::_1, tmpl::pin<Metavariables>>>;

    static constexpr bool use_alternative_parsing =
        std::is_same_v<alternative_parsable_types, tmpl::list<T...>>;
    static constexpr bool use_hybrid_parsing =
        not use_alternative_parsing and
        tmpl::size<alternative_parsable_types>::value > 1;

    if constexpr (use_alternative_parsing) {
      return Options_variant_detail::parse_as_alternatives<Result,
                                                           Metavariables>(
          options, alternative_parsable_types{});
    } else {
      Result result{};
      std::string errors{};
      bool constructed = false;

      if constexpr (use_hybrid_parsing) {
        try {
          result = Options_variant_detail::parse_as_alternatives<Result,
                                                                 Metavariables>(
              options, alternative_parsable_types{});
          constructed = true;
        } catch (const Options::detail::propagate_context& e) {
          // This alternative failed, but a later one may succeed.
          errors += "\n\n" + e.message();
        }
      }

      const auto try_parse = [&constructed, &errors, &options,
                              &result](auto alternative_v) {
        using Alternative = tmpl::type_from<decltype(alternative_v)>;
        if (use_hybrid_parsing and
            tmpl::any<alternative_parsable_types,
                      std::is_same<tmpl::_1, tmpl::pin<Alternative>>>::value) {
          return;
        }
        if (constructed) {
          return;
        }
        try {
          result = options.parse_as<Alternative, Metavariables>();
          constructed = true;
        } catch (const Options::detail::propagate_context& e) {
          // This alternative failed, but a later one may succeed.
          errors += "\n\n" + e.message();
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(try_parse(tmpl::type_<T>{}));
      if (not constructed) {
        try {
          options.parse_as<Options_detail::variant_parse_error<T...>,
                           Metavariables>();
        } catch (const Options::detail::propagate_context& e) {
          throw Options::detail::propagate_context(
              e.message() + "\n\nPossible errors:" + errors);
        }
      }
      return result;
    }
  }
};
}  // namespace Options

/// \cond
template <typename T, typename Metavariables>
struct YAML::convert<Options::Options_detail::CreateWrapper<T, Metavariables>> {
  static bool decode(
      const Node& node,
      Options::Options_detail::CreateWrapper<T, Metavariables>& rhs) {
    Options::Context context;
    context.top_level = false;
    context.append("While creating a " + pretty_type::name<T>());
    Options::Option options(node, std::move(context));
    rhs = Options::Options_detail::CreateWrapper<T, Metavariables>{
        Options::create_from_yaml<T>::template create<Metavariables>(options)};
    return true;
  }
};
/// \endcond

#include "Options/Factory.hpp"
