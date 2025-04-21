---
layout: default
title: Blog
permalink: /blog/
---

# Blog

Welcome to my writing space. Here I share notes, experiments, and tutorials on statistics, machine learning, quantitative finance, and data science from the field. I aim for clarity, practicality, and the occasional rabbit hole.

<!-- ## 📚 Series & Topics

- [Book Reviews](/tags/books/)

--- -->

## All Posts

---

<ul class="post-list">
  {% for post in site.posts %}
    <li>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      <span class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</span><br/>
      <p>{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
    </li>
  {% endfor %}
</ul>