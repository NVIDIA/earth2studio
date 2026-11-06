{{objname}}
{{ underline }}==============

.. currentmodule:: {{module}}

.. autoclass:: {{ objname }}

   {% block methods %}
    .. automethod:: __call__
    .. automethod:: set_rng
    {% for method in ["initialize", "step", "create_iterator", "forcing_coords"] %}
    {% if method in methods %}
    .. automethod:: {{ method }}
    {% endif %}
    {% endfor %}
    .. automethod:: input_coords
    .. automethod:: output_coords
    .. automethod:: default_sources
    .. automethod:: load_default_package
    .. automethod:: load_model
   {% endblock %}

.. _sphx_glr_backref_{{module}}.{{objname}}:

.. minigallery:: {{module}}.{{objname}}
      :add-heading:
      :heading-level: ^

.. raw:: html

    <div class="clearer"></div>
